# Python-bioformats is distributed under the GNU General Public
# License, but this file is licensed under the more permissive BSD
# license.  See the accompanying file LICENSE for details.
#
# Copyright (c) 2009-2014 Broad Institute
# All rights reserved.

'''Bioformats package - wrapper for loci.bioformats java code

'''

from __future__ import absolute_import, unicode_literals

try:
    from ._version import version as __version__
except ImportError:
    # We're running in a tree that doesn't have a _version.py, so we don't know what our version is.
    __version__ = "0.0.0"

import os.path
import javabridge
from . import formatreader as _formatreader
from . import formatwriter as _formatwriter

_jars_dir = os.path.join(os.path.dirname(__file__), 'jars')

JAR_VERSION = '6.5.1'

JARS = javabridge.JARS + [os.path.realpath(os.path.join(_jars_dir, name + '.jar'))
                          for name in ['bioformats_package']]
"""List of directories, jar files, and zip files that should be added
to the Java virtual machine's class path."""

# See http://www.loci.wisc.edu/software/bio-formats
READABLE_FORMATS = ('1sc', '2fl', 'acff', 'afi', 'afm', 'aim', 'al3d', 'ali',
                    'am', 'amiramesh', 'apl', 'arf', 'avi', 'bif', 'bin', 'bip',
                    'bmp', 'btf', 'c01', 'cfg', 'ch5', 'cif', 'cr2', 'crw', 
                    'cxd', 'czi', 'dat', 'dcm', 'dib', 'dicom', 'dm2', 'dm3',
                    'dm4', 'dti', 'dv', 'eps', 'epsi', 'exp', 'fdf', 'fff',
                    'ffr', 'fits', 'flex', 'fli', 'frm', 'gel', 'gif', 'grey',
                    'h5', 'hdf', 'hdr', 'hed', 'his', 'htd', 'html', 'hx', 'i2i',
                    'ics', 'ids', 'im3', 'img', 'ims', 'inr', 'ipl', 'ipm', 'ipw',
                    'j2k', 'jp2', 'jpeg', 'jpf', 'jpg', 'jpk', 'jpx', 'klb',
                    'l2d', 'labels', 'lei', 'lif', 'liff', 'lim', 'lms', 'lsm',
                    'map', 'mdb', 'mea', 'mnc', 'mng', 'mod', 'mov', 'mrc', 'mrcs',
                    'mrw', 'msr', 'mtb', 'mvd2', 'naf', 'nd', 'nd2', 'ndpi', 'ndpis',
                    'nef', 'nhdr', 'nii', 'nii.gz', 'nrrd', 'obf', 'obsep', 'oib',
                    'oif', 'oir', 'ome', 'ome.btf', 'ome.tf2', 'ome.tf8', 'ome.tif',
                    'ome.tiff', 'ome.xml', 'par', 'pbm', 'pcoraw', 'pcx', 'pds',
                    'pgm', 'pic', 'pict', 'png', 'pnl', 'ppm', 'pr3', 'ps', 'psd',
                    'qptiff', 'r3d', 'raw', 'rcpnl', 'rec', 'res', 'scn', 'sdt',
                    'seq', 'sif', 'sld', 'sm2', 'sm3', 'spc', 'spe', 'spi', 'st',
                    'stk', 'stp', 'svs', 'sxm', 'tc.', 'tf2', 'tf8', 'tfr', 'tga',
                    'tif', 'tiff', 'tnb', 'top', 'txt', 'v', 'vff', 'vms', 'vsi',
                    'vws', 'wat', 'wlz', 'wpi', 'xdce', 'xml', 'xqd', 'xqf', 'xv',
                    'xys', 'zfp', 'zfr', 'zvi')

WRITABLE_FORMATS = ('avi', 'eps', 'epsi', 'ics', 'ids', 'jp2', 'jpeg', 'jpg',
                    'mov', 'ome', 'ome.tiff', 'png', 'ps', 'tif', 'tiff')

OMETiffWriter = _formatwriter.make_ome_tiff_writer_class()
ChannelSeparator = _formatreader.make_reader_wrapper_class(
    "loci/formats/ChannelSeparator")


from .metadatatools import createOMEXMLMetadata as create_ome_xml_metadata
from .metadatatools import wrap_imetadata_object
from . import metadatatools as _metadatatools
PixelType = _metadatatools.make_pixel_type_class()
get_metadata_options = _metadatatools.get_metadata_options

# Reading images

ImageReader = _formatreader.ImageReader
load_image = _formatreader.load_using_bioformats
load_image_url = _formatreader.load_using_bioformats_url

# Cached image readers

get_image_reader = _formatreader.get_image_reader
release_image_reader = _formatreader.release_image_reader
clear_image_reader_cache = _formatreader.clear_image_reader_cache

# Metadata

from .omexml import OMEXML
get_omexml_metadata = _formatreader.get_omexml_metadata

# Writing images

write_image = _formatwriter.write_image

from .omexml import PT_UINT16, PT_UINT8, PT_BIT

# Omero

from .formatreader import use_omero_credentials, set_omero_credentials, get_omero_credentials
from .formatreader import set_omero_login_hook, omero_logout, has_omero_packages
from .formatreader import K_OMERO_SERVER, K_OMERO_PORT, K_OMERO_USER, K_OMERO_SESSION_ID,\
         K_OMERO_PASSWORD, K_OMERO_CONFIG_FILE

from . import omexml


if __name__ == "__main__":
    # Handy-dandy PyShell for exploring BioFormats / Rhino / ImageJ
    import wx.py.PyCrust

    wx.py.PyCrust.main()
    javabridge.kill_vm()
