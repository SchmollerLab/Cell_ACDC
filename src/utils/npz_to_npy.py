import os
import sys
import numpy as np

src_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
sys.path.append(src_path)

import prompts
filetypes = (
        ('npz files', '*.npz'),
        ('All files', '*.*')
    )
npz_paths = prompts.multi_files_dialog(title='Select one or more .npz files',
                                       filetypes=filetypes)

if not npz_paths:
    exit('Execution aborted')

npy_folder = prompts.folder_dialog(title='Select folder where to save .npy files')

for npz_path in npz_paths:
    filename, ext = os.path.splitext(os.path.basename(npz_path))
    if ext == '.npz':
        print('---------------------------------------')
        print(f'Loading: {npz_path}')
        a = np.load(npz_path)['arr_0']
        npy_path = os.path.join(npy_folder, f'{filename}.npy')
        print(f'Saving: {npy_path}')
        np.save(npy_path, a)
