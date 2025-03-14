import os
import shutil
import tempfile

from typing import Tuple

from tqdm import tqdm

import numpy as np
import h5py

from cellacdc import myutils, bioio_sample_data_folderpath

def removeInvalidCharacters(chName_in):
    # Remove invalid charachters
    chName = "".join(
        c if c.isalnum() or c=='_' or c=='' else '_' for c in chName_in
    )
    trim_ = chName.endswith('_')
    while trim_:
        chName = chName[:-1]
        trim_ = chName.endswith('_')

def getFilename(
        filenameNOext, s0p, appendTxt, series, ext, 
        return_basename=False
    ):
    # Do not allow dots in the filename since it breaks stuff here and there
    filenameNOext = filenameNOext.replace('.', '_')
    basename = f'{filenameNOext}_s{s0p}_'
    filename = f'{basename}{appendTxt}{ext}'
    if return_basename:
        return filename, basename
    else:
        return filename

def saveImgDataChannel(
        reader, 
        series: int, 
        images_path: os.PathLike, 
        filenameNOext: str, 
        s0p: str, 
        chName: str,
        ch_idx: int, 
        idxs: dict, 
        SizeT: int,
        SizeZ: int, 
        TimeIncrement: float,
        PhysicalSizeZ: float,
        PhysicalSizeY: float,
        PhysicalSizeX: float,
        to_h5: bool, 
        timeRangeToSave: Tuple[int, int],
    ):
    if to_h5:
        filename = getFilename(
            filenameNOext, s0p, chName, series, '.h5'
        )
        tempDir = tempfile.mkdtemp()
        tempFilepath = os.path.join(tempDir, filename)
        print('==========================================================')
        print(f'.h5 tempfile: "{tempFilepath}"')
        print('==========================================================')
        h5f = h5py.File(tempFilepath, 'w')
        # Read SizeX and SizeY from the shape of one image
        imgData = reader.read(
            c=ch_idx, z=0, t=0, series=series, rescale=False
        )
        shape = (SizeT, SizeZ, *imgData.shape)
        chunks = (1,1,*imgData.shape)
        imgData_ch = h5f.create_dataset(
            'data', shape, dtype=imgData.dtype,
            chunks=chunks, shuffle=False
        )
    else:
        filename = getFilename(
            filenameNOext, s0p, chName, series, '.tif'
        )
        imgData_ch = []

    framesRange = range(timeRangeToSave[0]-1, timeRangeToSave[1])
    filePath = os.path.join(images_path, filename)
    dimsIdx = {'c': ch_idx} 
    numFrames = len(framesRange)
    num_imgs = numFrames*SizeZ
    pbar = tqdm(
        total=num_imgs, 
        ncols=100, 
        desc=f'Reading image (z 0/{SizeZ}, t 0/{numFrames})'
    )
    for t in framesRange:
        imgData_z = []
        dimsIdx['t'] = t
        for z in range(SizeZ):
            pbar.set_description(
                f'Reading image (z {z+1}/{SizeZ}, t {t+1}/{numFrames})'
            )
            dimsIdx['z'] = z
            idx = None
            imgData = reader.read(
                c=ch_idx, z=z, t=t, series=series, rescale=False,
                index=idx
            )
            if to_h5:
                imgData_ch[t, z] = imgData
            else:
                imgData_z.append(imgData)
            
            pbar.update()

        if not to_h5:
            imgData_z = np.squeeze(np.array(imgData_z, dtype=imgData.dtype))
            imgData_ch.append(imgData_z)
    pbar.close()

    if not to_h5:
        imgData_ch = np.squeeze(np.array(imgData_ch, dtype=imgData.dtype))
        myutils.to_tiff(
            filePath, imgData_ch, 
            SizeT=SizeT,
            SizeZ=SizeZ,
            TimeIncrement=TimeIncrement,
            PhysicalSizeZ=PhysicalSizeZ,
            PhysicalSizeY=PhysicalSizeY,
            PhysicalSizeX=PhysicalSizeX,
        )
    else:
        h5f.close()
        shutil.move(tempFilepath, filePath)
        shutil.rmtree(tempDir)

def dump_exception(err, error_id):
    import pickle
    error_path = os.path.join(
        bioio_sample_data_folderpath, f'error_{error_id}.pkl'
    )
    with open(error_path, 'wb') as file:
        pickle.dump(err, file)

def check_raise_exception(error_id):
    import pickle
    error_path = os.path.join(
        bioio_sample_data_folderpath, f'error_{error_id}.pkl'
    )
    if not os.path.exists(error_path):
        return
    
    with open(error_path, "rb") as file:
        err = pickle.load(file)
    
    os.remove(error_path)
    
    raise err