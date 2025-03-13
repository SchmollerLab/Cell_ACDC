import os

from tqdm import tqdm

import numpy as np

from cellacdc import bioio_sample_data_folderpath
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

ap.add_argument(
    '-c', 
    '--SizeC', 
    required=True, 
    type=int, 
    metavar='SIZEC',
    help='Number of channels in the microscopy file.'
)

ap.add_argument(
    '-t', 
    '--SizeT', 
    required=True, 
    type=int, 
    metavar='SIZET',
    help='Number of timepoints in the microscopy file.'
)

ap.add_argument(
    '-z', 
    '--SizeZ', 
    required=True, 
    type=int, 
    metavar='SIZEZ',
    help='Number of z-slices in a single z-stack.'
)

args = vars(ap.parse_args())
raw_filepath = args['filepath']

SizeC = args['SizeC']
SizeT = args['SizeT']
SizeZ = args['SizeZ']

if SizeT >= 4:
    sampleSizeT = 4
else:
    sampleSizeT = SizeT 
if SizeZ > 20:
    sampleSizeZ = 20
else:
    sampleSizeZ = SizeZ

allChannelsData = []
with bioformats.ImageReader(raw_filepath) as reader:
    numIter = SizeC*sampleSizeT*sampleSizeZ
    pbar = tqdm(total=numIter, ncols=100, leave=False)
    
    for c in range(SizeC):
        imgData_tz = []
        for t in range(sampleSizeT):   
            imgData_z = []
            for z in range(sampleSizeZ):
                imgData = reader.read(c=c, z=z, t=t, rescale=False)
                imgData_z.append(imgData) 
                pbar.update()              
            imgData_z = np.array(imgData_z, dtype=imgData.dtype)
            imgData_z = np.squeeze(imgData_z)
            imgData_tz.append(imgData_z)
        imgData_tz = np.array(imgData_tz, dtype=imgData.dtype)
        imgData_tz = np.squeeze(imgData_tz)
        allChannelsData.append(imgData_tz)

os.makedirs(bioio_sample_data_folderpath, exist_ok=True)
for c, channel_data in enumerate(allChannelsData):
    filepath = os.path.join(
        bioio_sample_data_folderpath, f"sample_channel_{c}.npy"
    )
    np.save(filepath, channel_data)