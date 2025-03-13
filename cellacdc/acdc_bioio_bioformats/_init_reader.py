from cellacdc import acdc_bioio_bioformats as bioformats

import argparse

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

with bioformats.ImageReader(raw_filepath, qparent=None) as reader:
    print(reader)