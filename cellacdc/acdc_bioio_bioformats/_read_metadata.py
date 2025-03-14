import os
import pickle

from tqdm import tqdm

import numpy as np

from cellacdc import bioio_sample_data_folderpath
from cellacdc import acdc_bioio_bioformats as bioformats

import argparse

try:
    ap = argparse.ArgumentParser(
        prog='Cell-ACDC process', 
        description='Used to spawn a separate process', 
        formatter_class=argparse.RawTextHelpFormatter
    )

    ap.add_argument(
        '-f', 
        '--filepath', 
        required=True, 
        type=str, 
        metavar='FILEPATH',
        help='Filepath of a raw microscopy file to test.'
    )

    args = vars(ap.parse_args())
    raw_filepath = args['filepath']

    metadataXML = bioformats.get_omexml_metadata(raw_filepath)
    metadata = bioformats.OMEXML().init_from_metadata(metadataXML)

    print(metadata)

    os.makedirs(bioio_sample_data_folderpath, exist_ok=True)
    metadataXML_filepath = os.path.join(
        bioio_sample_data_folderpath, 'metadataXML.txt'
    )
    metadataXML.to_file(metadataXML_filepath)

    metadata_filepath = os.path.join(
        bioio_sample_data_folderpath, 'metadata.txt'
    )
    metadata.to_file(metadata_filepath)
except Exception as err:
    bioformats._utils.dump_exception(err)