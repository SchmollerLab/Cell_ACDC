import os
import sys
import numpy as np
import skimage.io
import shutil
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
        T, Y, X = sub_img.shape
        sub_img.shape = T, 1, 1, Y, X, 1  # imageJ format should always have TZCYXS data shape
        new_tif.save(sub_img, metadata=metadata)
    # if i > 0:
    #     leading_frame_path = os.path.join(dir_path,
    #                                 f'{i:02d}-{i+1:02d}_leading_frame.tif')
    #     leading_frame = img[start-1]
    #     with TiffWriter(leading_frame_path, imagej=True) as _tif:
    #         Y, X = leading_frame.shape
    #         leading_frame.shape = 1, 1, 1, Y, X, 1  # imageJ format should always have TZCYXS data shape
    #         _tif.save(leading_frame, metadata=metadata)

print('Moving original files into subfolder...')
basename_idx = filename.find('_phase_contr.tif')
basename = filename[:basename_idx]
segm_file_path = os.path.join(dir_path, f'{basename}_segm.npy')
align_shifts_file_path = os.path.join(dir_path, f'{basename}_align_shift.npy')
last_tracked_i_fp = os.path.join(dir_path, f'{basename}_last_tracked_i.txt')
phc_aligned_file_path = os.path.join(dir_path, f'{basename}_phc_aligned.npy')
move_files = [img_path]
if os.path.exists(segm_file_path):
    move_files.append(segm_file_path)
if os.path.exists(align_shifts_file_path):
    move_files.append(align_shifts_file_path)
if os.path.exists(last_tracked_i_fp):
    move_files.append(last_tracked_i_fp)
if os.path.exists(phc_aligned_file_path):
    move_files.append(phc_aligned_file_path)

dir_dst = os.path.join(dir_path, 'Original_unsplitted_files')
if not os.path.exists(dir_dst):
    os.mkdir(dir_dst)
for f in move_files:
    fn = os.path.basename(f)
    dst = os.path.join(dir_dst, fn)
    shutil.move(f, dst)


print('')
print(f'Done. Splitted images saved to {dir_path}')
print('')
print('-----------------------------------------')
