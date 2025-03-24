from cellacdc import acdc_bioio_bioformats as bioformats

import argparse

ap = bioformats._utils.setup_argparser()

try:
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
except Exception as err:
    args = vars(ap.parse_args())
    uuid4 = args['uuid']
    
    bioformats._utils.dump_exception(err, uuid4)