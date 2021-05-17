import os
import sys
import numpy as np
import skimage.io
from tifffile.tifffile import TiffWriter, TiffFile
from tqdm import tqdm

script_dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.dirname(script_dir_path)
sys.path.append(src_path)

from lib import file_dialog, single_entry_messagebox

print('-----------------------------------------')

img_path = file_dialog(title='Select image file to split')

num_splits = int(single_entry_messagebox(
                                     title='Number of splits',
                                     entry_label='Number of splits: ',
                                     input_txt='2',
                                     toplevel=False).entry_txt)

print('Loading data...')
with TiffFile(img_path) as tif:
    metadata = tif.imagej_metadata
    img = tif.asarray()

if img.ndim < 3:
    raise IndexError('2D Images cannot be further splitted.')
print('Data loaded.')
print('')

print('Splitting and saving images...')
filename = os.path.basename(img_path)
dir_path = os.path.dirname(img_path)
split_size = int(len(img)/num_splits)
d = int(np.log10(num_splits))+1
for i in tqdm(range(num_splits), unit=' img'):
    start = i*split_size
    end = start+split_size
    if end > len(img):
        end = len(img)
    if i == num_splits-1 and end < len(img):
        end = len(img)
    sub_img = img[start:end]
    sub_img_path = os.path.join(dir_path, f'{i+1:02d}_{filename}')
    with TiffWriter(sub_img_path, imagej=True) as new_tif:
        Z, Y, X = sub_img.shape
        sub_img.shape = 1, Z, 1, Y, X, 1  # imageJ format should always have TZCYXS data shape
        new_tif.save(sub_img, metadata=metadata)
print('')
print(f'Done. Splitted images saved to {dir_path}')
print('')
print('-----------------------------------------')
