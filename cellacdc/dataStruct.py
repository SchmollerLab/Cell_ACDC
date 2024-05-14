import sys
import os
import shutil
import re
import traceback
import time
import datetime
import tempfile
import h5py
import difflib
import pathlib
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from natsort import natsorted
from pprint import pprint
from functools import wraps, partial
from itertools import permutations

from qtpy.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QVBoxLayout, QPushButton, QLabel, QStyleFactory,
    QWidget, QMessageBox, QPlainTextEdit, QHBoxLayout
)
from qtpy.QtCore import (
    Qt, QObject, Signal, QThread, QMutex, QWaitCondition,
    QEventLoop
)
from qtpy import QtGui

# Here we use from cellacdc because this script is laucnhed in
# a separate process that doesn't have a parent package
from . import issues_url
from . import exception_handler
from . import qrc_resources
from . import apps, myutils, widgets, html_utils, printl
from . import load, settings_csv_path
from . import _palettes
from . import recentPaths_path, cellacdc_path, settings_folderpath
from . import urls
from . import acdc_fiji_path
from . import fiji_macros

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.cellacdc.pyqt.v1' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass

def worker_exception_handler(func):
    @wraps(func)
    def run(self):
        try:
            result = func(self)
        except Exception as error:
            result = None
            self.critical.emit(error)
        return result
    return run

class bioFormatsWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    progressPbar = Signal(int)
    initPbar = Signal(int)
    criticalError = Signal(str, str, str)
    filesExisting = Signal(str)
    confirmMetadata = Signal(
        str, float, str, int, int, int, int,
        float, str, float, float, float,
        str, list, list, str, str, object
    )
    critical = Signal(object)
    sigFinishedReadingSampleImageData = Signal(object)
    # aborted = Signal()

    def __init__(
            self, raw_src_path, rawFilenames, exp_dst_path,
            mutex, waitCond, rawDataStruct
        ):
        QObject.__init__(self)
        self.raw_src_path = raw_src_path
        self.exp_dst_path = exp_dst_path
        self.rawFilenames = rawFilenames
        self.mutex = mutex
        self.waitCond = waitCond
        self.askReplacePosFiles = True
        self.overWriteMetadata = False
        self.trustMetadataReader = False
        self.rawDataStruct = rawDataStruct
        self.overwritePos = False
        self.addFiles = False
        self.cancel = False
    
    def readSampleData(self, rawFilePath, SizeC, SizeT, SizeZ):
        sampleImgData = {}
        self.progress.emit('Reading sample image data...')
        dimsIdx = {}
        if SizeT >= 4:
            sampleSizeT = 4
        else:
            sampleSizeT = SizeT 
        if SizeZ > 20:
            sampleSizeZ = 20
        else:
            sampleSizeZ = SizeZ
        with bioformats.ImageReader(rawFilePath) as reader:
            permut_pbar = tqdm(total=6, ncols=100)
            for dimsOrd in permutations('zct', 3):
                allChannelsData = []
                idxs = self.buildIndexes(SizeC, SizeT, SizeZ, dimsOrd)
                numIter = SizeC*sampleSizeT*sampleSizeZ
                pbar = tqdm(total=numIter, ncols=100, leave=False)
                skipPermutation = False
                for c in range(SizeC):
                    dimsIdx['c'] = c
                    imgData_tz = []
                    for t in range(sampleSizeT):   
                        dimsIdx['t'] = t
                        imgData_z = []
                        for z in range(sampleSizeZ):
                            dimsIdx['z'] = z
                            try:
                                idx = self.getIndex(idxs, dimsIdx, dimsOrd)
                                imgData = reader.read(
                                    c=c, z=z, t=0, rescale=False, index=idx
                                )
                            except Exception as e:
                                skipPermutation = True
                                break
                            imgData_z.append(imgData) 
                            pbar.update()
                        if skipPermutation:
                            break               
                        imgData_z = np.array(imgData_z, dtype=imgData.dtype)
                        imgData_z = np.squeeze(imgData_z)
                        imgData_tz.append(imgData_z)
                    if not skipPermutation:
                        imgData_tz = np.array(imgData_tz, dtype=imgData.dtype)
                        imgData_tz = np.squeeze(imgData_tz)
                        allChannelsData.append(imgData_tz)
                pbar.close()
                permut_pbar.update(1)
                if not skipPermutation:
                    sampleImgData[''.join(dimsOrd)] = allChannelsData
            permut_pbar.close()
        self.sigFinishedReadingSampleImageData.emit(sampleImgData)
        return sampleImgData

    def getSizeZ(self, rawFilePath):
        try:
            if rawFilePath.endswith('.ome.tif'):
                metadata = load.OMEXML(rawFilePath)
                metadataXML = metadata.omexml_string
            else:
                metadataXML = bioformats.get_omexml_metadata(rawFilePath)
                metadata = bioformats.OMEXML(metadataXML)
            SizeZ = int(metadata.image().Pixels.SizeZ)
            return SizeZ
        except Exception as e:
            return self.SizeZ

    def readMetadata(self, raw_src_path, filename):
        rawFilePath = os.path.join(raw_src_path, filename)

        self.progress.emit('Reading OME metadata...')

        try:
            if rawFilePath.endswith('.ome.tif'):
                metadata = load.OMEXML(rawFilePath)
                metadataXML = metadata.omexml_string
            else:
                metadataXML = bioformats.get_omexml_metadata(rawFilePath)
                metadata = bioformats.OMEXML(metadataXML)
            self.metadata = metadata
            self.metadataXML = metadataXML
        except Exception as e:
            self.isCriticalError = True
            self.criticalError.emit(
                'reading image data or metadata',
                traceback.format_exc(), filename
            )
            return True

        try:
            LensNA = float(metadata.instrument().Objective.LensNA)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: LensNA not found in metadata.')
            self.progress.emit(
                '===================================================')
            LensNA = 1.4

        if self.rawDataStruct != 2:
            try:
                SizeS = int(metadata.get_image_count())
            except Exception as e:
                self.progress.emit(
                    '===================================================')
                self.progress.emit(rawFilePath)
                self.progress.emit('WARNING: SizeS not found in metadata.')
                self.progress.emit(
                    '===================================================')
                SizeS = 1
        else:
            SizeS = self.SizeS

        DimensionOrder = 'ztc'

        try:
            SizeZ = int(metadata.image().Pixels.SizeZ)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: SizeZ not found in metadata.')
            self.progress.emit(
                '===================================================')
            SizeZ = 1

        try:
            SizeT = int(metadata.image().Pixels.SizeT)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: SizeT not found in metadata.')
            self.progress.emit(
                '===================================================')
            SizeT = 1

        try:
            Pixels = metadata.image().Pixels
            TimeIncrement = float(Pixels.node.get('TimeIncrement'))
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: TimeIncrement not found in metadata.')
            self.progress.emit(
                '===================================================')
            TimeIncrement = 1.0

        try:
            Pixels = metadata.image().Pixels
            TimeIncrementUnit = Pixels.node.get('TimeIncrementUnit')
            if TimeIncrementUnit is None:
                raise
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: TimeIncrementUnit not found in metadata.')
            self.progress.emit(
                '===================================================')
            TimeIncrementUnit = 's'

        try:
            SizeC = int(metadata.image().Pixels.SizeC)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: SizeC not found in metadata.')
            self.progress.emit(
                '===================================================')
            SizeC = 1

        try:
            PhysicalSizeX = float(metadata.image().Pixels.PhysicalSizeX)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: PhysicalSizeX not found in metadata.')
            self.progress.emit(
                '===================================================')
            PhysicalSizeX = 1.0

        try:
            PhysicalSizeY = float(metadata.image().Pixels.PhysicalSizeY)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: PhysicalSizeY not found in metadata.')
            self.progress.emit(
                '===================================================')
            PhysicalSizeY = 1.0

        try:
            PhysicalSizeZ = float(metadata.image().Pixels.PhysicalSizeZ)
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: PhysicalSizeZ not found in metadata.')
            self.progress.emit(
                '===================================================')
            PhysicalSizeZ = 1.0

        try:
            Pixels = metadata.image().Pixels
            PhysicalSizeUnit = Pixels.node.get('PhysicalSizeXUnit')
            if PhysicalSizeUnit is None:
                raise
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: PhysicalSizeUnit not found in metadata.')
            self.progress.emit(
                '===================================================')
            PhysicalSizeUnit = 'μm'

        try:
            ImageName = metadata.image().Name
            if ImageName is None:
                raise
        except Exception as e:
            self.progress.emit(
                '===================================================')
            self.progress.emit(rawFilePath)
            self.progress.emit('WARNING: Image Name not found in metadata.')
            self.progress.emit(
                '===================================================')
            ImageName = ''


        if self.rawDataStruct != 2:
            try:
                chNames = ['']*SizeC
                for c in range(SizeC):
                    try:
                        chNames[c] = metadata.image().Pixels.Channel(c).Name
                    except Exception as e:
                        pass
            except Exception as e:
                self.progress.emit(
                    '===================================================')
                self.progress.emit(rawFilePath)
                self.progress.emit('WARNING: chNames not found in metadata.')
                self.progress.emit(
                    '===================================================')
                chNames = ['']*SizeC
        else:
            chNames = self.chNames
            SizeC = len(self.chNames)

        if self.rawDataStruct != 2:
            try:
                emWavelens = [500.0]*SizeC
                for c in range(SizeC):
                    try:
                        Channel = metadata.image().Pixels.Channel(c)
                        emWavelen = Channel.node.get("EmissionWavelength")
                        emWavelens[c] = float(emWavelen)
                    except Exception as e:
                        emWavelens[c] = 0.0
            except Exception as e:
                traceback.print_exc()
                self.progress.emit(
                    '===================================================')
                self.progress.emit(rawFilePath)
                self.progress.emit('WARNING: EmissionWavelength not found in metadata.')
                self.progress.emit(
                    '===================================================')
                emWavelens = [500.0]*SizeC
        else:
            emWavelens = [500.0]*SizeC

        if self.trustMetadataReader:
            self.LensNA = LensNA
            self.DimensionOrder = DimensionOrder
            self.SizeT = SizeT
            self.SizeZ = SizeZ
            self.SizeC = SizeC
            self.SizeS = SizeS
            self.TimeIncrement = TimeIncrement
            self.PhysicalSizeX = PhysicalSizeX
            self.PhysicalSizeY = PhysicalSizeY
            self.PhysicalSizeZ = PhysicalSizeZ
            # self.chNames = chNames
            self.emWavelens = emWavelens
        else:
            sampleImgData = None
            while True:
                self.mutex.lock()
                if self.rawDataStruct != 2:
                    sampleImgData = self.readSampleData(
                        rawFilePath, SizeC, SizeT, SizeZ
                    )
                self.confirmMetadata.emit(
                    filename, LensNA, DimensionOrder, SizeT, SizeZ, SizeC, SizeS,
                    TimeIncrement, TimeIncrementUnit, PhysicalSizeX, PhysicalSizeY,
                    PhysicalSizeZ, PhysicalSizeUnit, chNames, emWavelens, ImageName,
                    rawFilePath, sampleImgData
                )
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()

                if self.metadataWin.cancel:
                    return True

                if not self.metadataWin.requestedReadingSampleImageDataAgain:
                    break
                LensNA = self.metadataWin.LensNA
                DimensionOrder = self.metadataWin.DimensionOrder
                SizeT = self.metadataWin.SizeT
                SizeZ = self.metadataWin.SizeZ
                SizeC = self.metadataWin.SizeC
                SizeS = self.metadataWin.SizeS
                TimeIncrement = self.metadataWin.TimeIncrement
                PhysicalSizeX = self.metadataWin.PhysicalSizeX
                PhysicalSizeY = self.metadataWin.PhysicalSizeY
                PhysicalSizeZ = self.metadataWin.PhysicalSizeZ
                chNames = self.metadataWin.chNames
                emWavelens = self.metadataWin.emWavelens

            if self.metadataWin.cancel:
                return True
            elif self.metadataWin.overWrite:
                self.overWriteMetadata = True
            elif self.metadataWin.trust:
                self.trustMetadataReader = True

            self.to_h5 = self.metadataWin.to_h5
            self.LensNA = self.metadataWin.LensNA
            self.DimensionOrder = self.metadataWin.DimensionOrder
            self.SizeT = self.metadataWin.SizeT
            self.SizeZ = self.metadataWin.SizeZ
            self.SizeC = self.metadataWin.SizeC
            self.SizeS = self.metadataWin.SizeS
            self.TimeIncrement = self.metadataWin.TimeIncrement
            self.PhysicalSizeX = self.metadataWin.PhysicalSizeX
            self.PhysicalSizeY = self.metadataWin.PhysicalSizeY
            self.PhysicalSizeZ = self.metadataWin.PhysicalSizeZ
            self.selectedPos = self.metadataWin.selectedPos
            self.chNames = self.metadataWin.chNames
            self.saveChannels = self.metadataWin.saveChannels
            self.emWavelens = self.metadataWin.emWavelens
            self.addImageName = self.metadataWin.addImageName

    def saveToPosFolder(
            self, p, raw_src_path, exp_dst_path, filename, series, p_idx=0
        ):
        rawFilePath = os.path.join(raw_src_path, filename)

        if os.path.basename(raw_src_path) == 'raw_microscopy_files':
            raw_src_path = os.path.dirname(raw_src_path)
        
        pos_name = f'Position_{p+1}'
        savePos = (
            'All Positions' in self.selectedPos or pos_name in self.selectedPos
        )
        if not savePos:
            return False

        pos_path = os.path.join(exp_dst_path, f'Position_{p+1}')
        images_path = os.path.join(pos_path, 'Images')

        if os.path.exists(images_path) and self.askReplacePosFiles:
            self.askReplacePosFiles = False
            self.mutex.lock()
            self.filesExisting.emit(pos_path)
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()
            if self.cancel:
                return True

        if os.path.exists(images_path) and self.overwritePos:
            shutil.rmtree(images_path)
        
        if not os.path.exists(images_path):
            os.makedirs(images_path, exist_ok=True)

        self.saveData(
            images_path, rawFilePath, filename, p, series, p_idx=p_idx
        )

        return False

    def removeInvalidCharacters(self, chName_in):
        # Remove invalid charachters
        chName = "".join(
            c if c.isalnum() or c=='_' or c=='' else '_' for c in chName_in
        )
        trim_ = chName.endswith('_')
        while trim_:
            chName = chName[:-1]
            trim_ = chName.endswith('_')

    def getFilename(
            self, filenameNOext, s0p, appendTxt, series, ext, 
            return_basename=False
        ):
        # Do not allow dots in the filename since it breaks stuff here and there
        filenameNOext = filenameNOext.replace('.', '_')
        if self.addImageName:
            try:
                ImageName = self.metadata.image(index=series).Name
                if not isinstance(ImageName, str):
                    raise
            except Exception as e:
                ImageName = ''
            self.removeInvalidCharacters(ImageName)
            basename = f'{filenameNOext}_{ImageName}_s{s0p}_'
            filename = f'{basename}{appendTxt}{ext}'
        else:
            basename = f'{filenameNOext}_s{s0p}_'
            filename = f'{basename}{appendTxt}{ext}'
        if return_basename:
            return filename, basename
        else:
            return filename
    
    def buildIndexes(self, SizeC, SizeT, SizeZ, DimensionOrder):
        SizesCTZ = {'c': SizeC, 't': SizeT, 'z': SizeZ}
        idxs = {}
        k_key, i_key, j_key = DimensionOrder
        idx = 0

        for k in range(SizesCTZ[k_key]):
            for i in range(SizesCTZ[i_key]):
                for j in range(SizesCTZ[j_key]):
                    idxs[(k,i,j)] = idx
                    idx += 1
        return idxs

    def getIndex(self, idxs, dimsIdx, DimensionOrder):
        """Get the index of the single 2D image given `dimsIdx`. 
        
        Note that `idxs` is generated in `buildIndexes` method.

        Example:
            Given a `DimensionOrder = 'tcz'`, and a 
            `dimsIdx = {'t': 0, 'c': 1, 'z': 0}` the returned index is `3`.

        Args:
            idxs (dict): Dictionary where the keys are tuples of `(t, c, z)`, 
                i.e., `t` is the index requested in the time dimension.
                Values are the corresponding incremental indexes in the correct
                order (see `DimensionOrder`)
            dimsIdx (dict): Dictionary with three keys, 'z', 't', and 'c'. 
                Values are the corresponding requested index.
            DimensionOrder (str): String of three lower-case characters 
                (no spaces, no punctuation) from a combination of 'z', 't', 
                and 'c'. This value determines the order of dimensions in the 
                Bio-Formats file. 

        Returns:
            int: incremental index requested.
        

        """
        dims = tuple([dimsIdx.get(v, 0) for v in DimensionOrder])
        return idxs[dims]
            
    def saveImgDataChannel(
            self, reader, series, images_path, filenameNOext, s0p, chName,
            ch_idx, idxs, SizeZ
        ):
        if self.to_h5:
            filename = self.getFilename(
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
            shape = (self.SizeT, self.SizeZ, *imgData.shape)
            chunks = (1,1,*imgData.shape)
            imgData_ch = h5f.create_dataset(
                'data', shape, dtype=imgData.dtype,
                chunks=chunks, shuffle=False
            )
        else:
            filename = self.getFilename(
                filenameNOext, s0p, chName, series, '.tif'
            )
            imgData_ch = []

        filePath = os.path.join(images_path, filename)
        dimsIdx = {'c': ch_idx} 
        for t in range(self.SizeT):
            imgData_z = []
            dimsIdx['t'] = t
            for z in range(SizeZ):
                dimsIdx['z'] = z
                if self.rawDataStruct != 2:
                    idx = self.getIndex(idxs, dimsIdx, self.DimensionOrder)
                else:
                    idx = None
                imgData = reader.read(
                    c=ch_idx, z=z, t=t, series=series, rescale=False,
                    index=idx
                )
                if self.to_h5:
                    imgData_ch[t, z] = imgData
                else:
                    imgData_z.append(imgData)

            if not self.to_h5:
                imgData_z = np.squeeze(np.array(imgData_z, dtype=imgData.dtype))
                imgData_ch.append(imgData_z)

        if not self.to_h5:
            
            imgData_ch = np.squeeze(np.array(imgData_ch, dtype=imgData.dtype))
            myutils.to_tiff(
                filePath, imgData_ch, 
                SizeT=self.SizeT,
                SizeZ=self.SizeZ,
                TimeIncrement=self.TimeIncrement,
                PhysicalSizeZ=self.PhysicalSizeZ,
                PhysicalSizeY=self.PhysicalSizeY,
                PhysicalSizeX=self.PhysicalSizeX,
            )
        else:
            h5f.close()
            shutil.move(tempFilepath, filePath)
            shutil.rmtree(tempDir)

    def saveData(self, images_path, rawFilePath, filename, p, series, p_idx=0):
        s0p = str(p+1).zfill(self.numPosDigits)
        self.progress.emit(
            f'Position {p+1}/{self.numPos}: saving data to {images_path}...'
        )
        filenameNOext, ext = os.path.splitext(filename)

        metadataXML_path = os.path.join(
            images_path,
            self.getFilename(filenameNOext, s0p, 'metadataXML', series, '.txt')
        )
        with open(metadataXML_path, 'w', encoding="utf-8") as txt:
            txt.write(self.metadataXML)

        metadata_filename, basename = self.getFilename(
            filenameNOext, s0p, 'metadata', series, '.csv', 
            return_basename=True
        )
        metadata_csv_path = os.path.join(images_path, metadata_filename)
        df = pd.DataFrame({
            'LensNA': self.LensNA,
            'DimensionOrder': self.DimensionOrder,
            'SizeT': self.SizeT,
            'SizeZ': self.SizeZ,
            'TimeIncrement': self.TimeIncrement,
            'PhysicalSizeZ': self.PhysicalSizeZ,
            'PhysicalSizeY': self.PhysicalSizeY,
            'PhysicalSizeX': self.PhysicalSizeX,
            'basename': basename
        }, index=['values']).T
        df.index.name = 'Description'

        ch_metadata = [
            chName for c, chName in enumerate(self.chNames)
            if self.saveChannels[c]
        ]
        description = [
            f'channel_{c}_name' for c in range(self.SizeC) 
            if self.saveChannels[c]
        ]
        ch_metadata.extend([
            wavelen for c, wavelen in enumerate(self.emWavelens)
            if self.saveChannels[c]
        ])
        description.extend([
            f'channel_{c}_emWavelen' for c in range(self.SizeC)
            if self.saveChannels[c]
        ])

        df_channelNames = pd.DataFrame({
            'Description': description,
            'values': ch_metadata
        }).set_index('Description')

        df = pd.concat([df, df_channelNames])

        if os.path.exists(metadata_csv_path):
            # Keep channel names already existing and not saved now
            existing_df = pd.read_csv(metadata_csv_path).set_index('Description')
            for c, chName in enumerate(self.chNames):
                if self.saveChannels[c]:
                    continue
                chName_idx = f'channel_{c}_name'
                chWavelen_idx = f'channel_{c}_emWavelen'
                try:
                    existing_chName = existing_df.at[chName_idx, 'values']
                    df.at[chName_idx, 'values'] = existing_chName
                except Exception as e:
                    traceback.print_exc()
                    pass
                
                try:
                    existing_chWavelen = existing_df.at[chWavelen_idx, 'values']
                    df.at[chWavelen_idx, 'values'] = existing_chWavelen
                except Exception as e:
                    traceback.print_exc()
                    pass

        df.to_csv(metadata_csv_path)

        idxs = self.buildIndexes(
            self.SizeC, self.SizeT, self.SizeZ, self.DimensionOrder
        )
        if self.rawDataStruct != 2:       
            SizeZ = self.getSizeZ(rawFilePath)
            with bioformats.ImageReader(rawFilePath) as reader:
                iter = enumerate(zip(self.chNames, self.saveChannels))
                for c, (chName, saveCh) in iter:
                    self.progressPbar.emit(1)
                    if not saveCh:
                        continue

                    self.progress.emit(
                        f'  Saving channel {c+1}/{len(self.chNames)} ({chName})'
                    )
                    self.saveImgDataChannel(
                        reader, series, images_path, filenameNOext, s0p,
                        chName, c, idxs, SizeZ
                    )

        elif self.rawDataStruct == 2:
            iter = enumerate(zip(self.chNames, self.saveChannels))
            pos_rawFilenames = []
            basename = filename
            for c, (chName, saveCh) in iter:
                self.progressPbar.emit(1)
                if not saveCh:
                    continue

                rawFilename = f'{basename}{p+1}_{chName}'
                pos_rawFilenames.append(rawFilename)
                raw_src_path = os.path.dirname(rawFilePath)
                rawFilePath = [
                    os.path.join(raw_src_path, f)
                    for f in myutils.listdir(raw_src_path)
                    if f.find(rawFilename)!=-1
                ][0]

                SizeZ = self.getSizeZ(rawFilePath)
                with bioformats.ImageReader(rawFilePath) as reader:
                    self.progress.emit(
                        f'  Saving channel {c+1}/{len(self.chNames)} ({chName})'
                    )
                    imgData_ch = []
                    self.saveImgDataChannel(
                        reader, series, images_path, filenameNOext, s0p,
                        chName, 0, idxs, SizeZ
                    )

            if self.moveOtherFiles or self.copyOtherFiles:
                # Move the other files present in the folder if they
                # contain "otherFilename" in the name
                otherFilename = f'{basename}{p+1}'
                rawFilePath = set()
                for f in myutils.listdir(raw_src_path):
                    notRawFile = all(
                        [f.find(rawName)==-1 for rawName in pos_rawFilenames]
                    )
                    isPosFile = f.find(otherFilename)!=-1
                    if isPosFile and notRawFile:
                        rawFilePath.add(os.path.join(raw_src_path, f))

                for file in rawFilePath:
                    # Determine basename, posNum and chName to build
                    # filename as "basename_s01_chName.ext"
                    _filename = os.path.basename(file)
                    m = re.findall(fr'{basename}(\d+)_(.+)', _filename)
                    if not m or len(m[0])!=2:
                        dst = os.path.join(images_path, _filename)
                    else:
                        _chNameWithExt = m[0][1]
                        _filename = f'{filenameNOext}_s{s0p}_{_chNameWithExt}'
                        dst = os.path.join(images_path, _filename)
                    if self.moveOtherFiles:
                        try:
                            shutil.move(file, dst)
                        except Exception as e:
                            self.progress.emit(e)
                    elif self.copyOtherFiles:
                        try:
                            shutil.copy(file, dst)
                        except Exception as e:
                            self.progress.emit(e)

    @worker_exception_handler
    def run(self):
        raw_src_path = self.raw_src_path
        exp_dst_path = self.exp_dst_path
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
        # bioformats.init_logger()
        self.progress.emit('Java VM running.')
        self.aborted = False
        self.isCriticalError = False
        for p, filename in enumerate(self.rawFilenames):
            if self.rawDataStruct == 0:
                if not self.overWriteMetadata:
                    abort = self.readMetadata(raw_src_path, filename)
                    if abort:
                        self.aborted = True
                        break

                self.numPos = self.SizeS
                self.numPosDigits = len(str(self.numPos))
                if p == 0:
                    self.initPbar.emit(self.numPos*self.SizeC)
                for p in range(self.SizeS):
                    abort = self.saveToPosFolder(
                        p, raw_src_path, exp_dst_path, filename, p
                    )
                    if abort:
                        self.aborted = True
                        break

            elif self.rawDataStruct == 1:
                if not self.overWriteMetadata:
                    abort = self.readMetadata(raw_src_path, filename)
                    if abort:
                        self.aborted = True
                        break
                self.numPos = len(self.rawFilenames)
                self.numPosDigits = len(str(self.numPos))
                if p == 0:
                    self.initPbar.emit(self.numPos*self.SizeC)
                abort = self.saveToPosFolder(
                    p, raw_src_path, exp_dst_path, filename, 0
                )
                if abort:
                    self.aborted = True
                    break

            else:
                break

            # Move files to raw_microscopy_files folder
            foldername = os.path.basename(self.raw_src_path)
            if foldername != 'raw_microscopy_files' and not self.aborted:
                rawFilePath = os.path.join(self.raw_src_path, filename)
                raw_path = os.path.join(raw_src_path, 'raw_microscopy_files')
                if not os.path.exists(raw_path):
                    os.mkdir(raw_path)
                dst = os.path.join(raw_path, filename)
                try:
                    shutil.move(rawFilePath, dst)
                except PermissionError as e:
                    self.progress.emit(e)

        if self.rawDataStruct == 2:
            filename = self.rawFilenames[0]
            if not self.overWriteMetadata:
                abort = self.readMetadata(raw_src_path, filename)
                if abort:
                    self.aborted = True
                    javabridge.kill_vm()
                    self.finished.emit()
                    return

            self.numPos = len(self.posNums)
            self.numPosDigits = len(str(self.numPos))
            self.initPbar.emit(self.numPos*self.SizeC)
            for p_idx, pos in enumerate(self.posNums):
                p = pos-1
                abort = self.saveToPosFolder(
                    p, raw_src_path, exp_dst_path, self.basename,
                    0, p_idx=p_idx
                )
                if abort:
                    self.aborted = True
                    break

            for filename in self.rawFilenames:
                # Move files to raw_microscopy_files folder
                foldername = os.path.basename(self.raw_src_path)
                if foldername != 'raw_microscopy_files' and not self.aborted:
                    rawFilePath = os.path.join(self.raw_src_path, filename)
                    raw_path = os.path.join(raw_src_path, 'raw_microscopy_files')
                    if not os.path.exists(raw_path):
                        os.mkdir(raw_path)
                    dst = os.path.join(raw_path, filename)
                    try:
                        shutil.move(rawFilePath, dst)
                    except PermissionError as e:
                        self.progress.emit(e)

        javabridge.kill_vm()
        self.finished.emit()
        

class createDataStructWin(QMainWindow):
    def __init__(
            self, parent=None, allowExit=False, buttonToRestore=None, 
            mainWin=None, start_JVM=True, version=None
        ):
        super().__init__(parent)

        self._version = version

        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module='dataStruct'
        )
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path

        if self._version is not None:
            logger.info(f'Initializing Data structure module v{self._version}...')
        else:
            logger.info(f'Initializing Data structure module...')

        is_linux = sys.platform.startswith('linux')
        is_mac = sys.platform == 'darwin'
        is_win = sys.platform.startswith("win")

        self.start_JVM = start_JVM
        self.allowExit = allowExit
        self.processFinished = False
        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin
        self.metadataDialogIsOpen = False
        self.df_settings = pd.read_csv(
            settings_csv_path, index_col='setting'
        )
        if 'lastDimensionOrder' in self.df_settings.index:
            val = self.df_settings.at['lastDimensionOrder', 'value']
            self.lastDimensionOrder = val

        self.setWindowTitle("Cell-ACDC - From raw microscopy file to tifs")
        self.setWindowIcon(QtGui.QIcon(":icon.ico"))

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QVBoxLayout()

        label = QLabel(
            'Creating data structure from raw microscopy file(s)...'
        )

        label.setStyleSheet("padding:5px 10px 10px 10px;")
        label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPixelSize(14)
        font.setBold(True)
        label.setFont(font)
        mainLayout.addWidget(label)

        informativeHtml = (
        """
        <html>
        <head>
        <title></title>
        <style type="text/css">
        blockquote {
         margin: 5;
         padding: 0;
        }
        </style>
        </head>
        <body>
        <blockquote>
        <p style="font-size:14px; line-height:1.2">
            This <b>wizard</b> will guide you through the <b>creation of the required
            data structure</b><br> starting from the raw microscopy file(s)
        </p>
        <p style="font-size:12px; line-height:1.2">
            Follow the instructions in the pop-up windows.<br>
            Note that pop-ups might be minimized or behind other open windows.<br>
            Keep an eye on the terminal/console in case of any error.
        </p>
        <p style="font-size:12px; line-height:1.2">
            Progress will be displayed below.
        </p>
        </blockquote>
        </body>
        </html>
        """
        )

        informativeText = QLabel(self)

        informativeText.setTextFormat(Qt.RichText)
        informativeText.setText(informativeHtml)
        informativeText.setStyleSheet("padding:5px 0px 10px 0px;")
        mainLayout.addWidget(informativeText)

        self.logWin = QPlainTextEdit()
        self.logWin.setReadOnly(True)
        mainLayout.addWidget(self.logWin)

        abortButton = widgets.cancelPushButton(' Abort process ')
        abortButton.clicked.connect(self.close)
        
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(abortButton)
        
        mainLayout.addLayout(buttonsLayout)

        mainLayout.setContentsMargins(20, 0, 20, 20)
        mainContainer.setLayout(mainLayout)

        self.mainLayout = mainLayout

        if not is_win and not is_mac:
            if parent is None:
                self.show()
            self.criticalOSnotSupported()
            self.close()
            raise OSError('This module is supported ONLY on Windows 10/10 and macOS')

        success, jar_dst_path = myutils.download_bioformats_jar(
            qparent=self, logger_info=self.logger.info, 
            logger_exception=self.logger.exception
        )
        global bioformats, javabridge
        self.logger.info('Checking if Java is installed...')
        myutils.check_upgrade_javabridge()
        try:
            import javabridge
        except ModuleNotFoundError as e:
            print('======================================')
            traceback_str = traceback.format_exc()
            self.logger.exception(traceback_str)
            print('======================================')
            cancel = myutils.install_javabridge_help(parent=self)
            if cancel:
                raise ModuleNotFoundError(
                    'User aborted javabridge installation'
                )

            isGitInstalled = myutils.check_git_installed(parent=self)
            if not isGitInstalled:
                raise ModuleNotFoundError(
                    'Git is not installed. Install from '
                    'https://git-scm.com/book/en/v2/Getting-Started-Installing-Git'
                )

            try:
                jre_path, jdk_path, url = myutils.download_java()
            except Exception as e:
                print('======================================')
                traceback_str = traceback.format_exc()
                self.logger.exception(traceback_str)
                print('======================================')
                java_info = myutils.get_java_url()
                url, file_size, os_foldername, unzipped_foldername = java_info
                acdc_java_path, _ = myutils.get_acdc_java_path()
                java_href = f'<a href="{url}">this</a>'
                s = (
                    f'1. Download {java_href} .zip file and unzip it.<br>'
                    '2. Inside the unzipped folder there should be a folder called '
                    f'"{unzipped_foldername}". Open that folder and copy its '
                    'content to the following path:<br><br>'
                    f'{os.path.join(acdc_java_path, os_foldername)}'
                )
                note = (
                    '<br><br><i>NOTE: if clicking on the link above does not work '
                    'copy the link below and paste it into the browser</i><br><br>'
                    f'{url}'
                )
                msg = widgets.myMessageBox(wrapText=False)
                txt = html_utils.paragraph(f"""
                    This module requires Java to work.<br><br>
                    Follow the instructions below and then try
                    launching this module again.<br><br>
                    {s}{note}
                """)
                msg.warning(self, 'Java not found', txt)
                
                err = s.replace('<br>', ' ')
                err = err.replace('<a href=', '')
                err = err.replace('>this</a>', '')
                raise ModuleNotFoundError(
                    'Installation of module "javabridge" failed. '
                    f'{err}'
                )

            if not is_win:
                cancel = myutils.install_java()
                if cancel:
                    raise ModuleNotFoundError(
                        'User aborted Java installation'
                    )
                    return

            myutils.install_javabridge()

        except Exception as e:
            print('======================================')
            traceback_str = traceback.format_exc()
            self.logger.exception(traceback_str)
            print('======================================')
            cancel = myutils.install_java()
            if cancel:
                raise ModuleNotFoundError(
                    'User aborted Java installation'
                )
                return
            myutils.install_javabridge(
                force_compile=True, attempt_uninstall_first=True
            )

        try:
            import javabridge
            from cellacdc import bioformats
        except Exception as e:
            print('===============================================================')
            traceback_str = traceback.format_exc()
            self.logger.exception(traceback_str)
            error_msg = (
                'Error while importing "javabridge" and "bioformats".\n\n'
                f'Please report error here: {issues_url}\n'
            )
            print(error_msg)
            print('===============================================================')

            title = 'Import javabridge/bioformats error'
            txt = error_msg.replace('\n', '<br>')
            txt = txt.replace(
                issues_url, html_utils.href_tag(issues_url, issues_url)
            )
            txt = html_utils.paragraph(txt)
            msg = widgets.myMessageBox(wrapText=False)
            msg.critical(
                self, title, txt, detailsText=traceback_str
            )
            raise ModuleNotFoundError(
                'Error when importing javabridge. See above for details.'
            )

    def criticalOSnotSupported(self):
        from cellacdc import widgets
        if self.parent() is None:
            msg = widgets.myMessageBox(self)
        else:
            msg = widgets.myMessageBox(self.parent())
        msg.setIcon(iconName='SP_MessageBoxCritical')
        msg.setWindowTitle('Not a supported OS')
        msg.addButton('    Ok     ')
        err_msg = (f"""
        <p style="font-size:12px">
        Unfortunately, the module "0. Create data structure from microscopy file(s)"
        is functional <b>only on Windows 10/11 and macOS</b>.<br><br>
        We are working on extending support to other Operating Systems.<br><br>
        Please open an issue on our
        <a href="https://github.com/SchmollerLab/Cell_ACDC/issues">
            GitHub page
        </a> to request this feature.<br>
        Note that <b>all other modules are functional</b> on
        macOS, Linux and Windows.<br><br>
        In the meantine, to create the required data structure,
        you can use an <b>automated Fiji macro</b> that you can download from
        <a href="https://github.com/SchmollerLab/Cell_ACDC/tree/main/FijiMacros">
            here
        </a>.<br><br>
        Check out the <b>instructions</b> on how to use the macros
        in the section  <b>"Create data structure using Fiji Macros"</b> of the
        user manual.<br><br>
        You can download the user manual from
        <a href="https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf">
            here
        </a>.
        </p>
        """)
        msg.addText(err_msg)
        # msg_label = msg.findChild(QLabel, "qt_msgbox_label")
        # msg_label.setOpenExternalLinks(False)
        # msg_label.linkActivated.connect(self.on_linkActivated)
        msg.exec_()


    def on_linkActivated(self, link):
        if link == 'manual':
            systems = {
                'nt': os.startfile,
                'posix': lambda foldername: os.system('xdg-open "%s"' % foldername),
                'os2': lambda foldername: os.system('open "%s"' % foldername)
                 }

            main_path = pathlib.Path(__file__).resolve().parents[1]
            userManual_path = main_path / 'UserManual'
            systems.get(os.name, os.startfile)(userManual_path)
        elif link == 'fiji':
            systems = {
                'nt': os.startfile,
                'posix': lambda foldername: os.system('xdg-open "%s"' % foldername),
                'os2': lambda foldername: os.system('open "%s"' % foldername)
                 }

            main_path = pathlib.Path(__file__).resolve().parents[1]
            fijiMacros_path = main_path / 'FijiMacros'
            systems.get(os.name, os.startfile)(fijiMacros_path)


    def getMostRecentPath(self):
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            if 'opened_last_on' in df.columns:
                df = df.sort_values('opened_last_on', ascending=False)
            self.MostRecentPath = df.iloc[0]['path']
            if not isinstance(self.MostRecentPath, str):
                self.MostRecentPath = ''
        else:
            self.MostRecentPath = ''

    def addToRecentPaths(self, raw_src_path):
        if not os.path.exists(raw_src_path):
            return
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            recentPaths = df['path'].to_list()
            if 'opened_last_on' in df.columns:
                openedOn = df['opened_last_on'].to_list()
            else:
                openedOn = [np.nan]*len(recentPaths)
            if raw_src_path in recentPaths:
                pop_idx = recentPaths.index(raw_src_path)
                recentPaths.pop(pop_idx)
                openedOn.pop(pop_idx)
            recentPaths.insert(0, raw_src_path)
            openedOn.insert(0, datetime.datetime.now())
            # Keep max 20 recent paths
            if len(recentPaths) > 20:
                recentPaths.pop(-1)
                openedOn.pop(-1)
        else:
            recentPaths = [raw_src_path]
            openedOn = [datetime.datetime.now()]
        df = pd.DataFrame({'path': recentPaths,
                           'opened_last_on': pd.Series(openedOn,
                                                       dtype='datetime64[ns]')})
        df.index.name = 'index'
        df.to_csv(recentPaths_path)

    @exception_handler
    def main(self):
        self.log('Asking how raw data is structured...')
        rawDataStruct, abort = self.askRawDataStruct()
        if abort:
            self.close()
            return

        self.rawDataStruct = rawDataStruct

        if rawDataStruct == 3:
            self.instructManualStruct()
            self.close()
            return

        self.log('Instructing to move raw data...')
        proceed = self.instructMoveRawFiles()
        if not proceed:
            self.close()
            return

        self.log(
            'Asking to select the folder that contains the microscopy files...'
        )
        self.getMostRecentPath()
        raw_src_path = QFileDialog.getExistingDirectory(
            self, 'Select folder containing the microscopy files', self.MostRecentPath)
        self.addToRecentPaths(raw_src_path)

        if raw_src_path == '':
            self.close()
            return

        self.log(
            'Checking file format of loaded files...'
        )
        rawFilenames = self.checkFileFormat(raw_src_path)
        if not rawFilenames:
            self.close()
            return

        if rawDataStruct == 2:
            proceed = self.attemptSeparateMultiChannel(rawFilenames)
            if not proceed:
                self.close()
                return

        self.log(
            'Asking in which folder to save the images files...'
        )
        exp_dst_path = QFileDialog.getExistingDirectory(
            self, 'Select the folder in which to save the images files',
            raw_src_path
        )
        if not exp_dst_path:
            self.close()
            return

        self.addToRecentPaths(exp_dst_path)

        self.log(
            'Starting a Java Virtual Machine...'
        )

        self.addPbar()

        # Set up separate thread for bioFormatsWorker class
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.thread = QThread()
        self.worker = bioFormatsWorker(
            raw_src_path, rawFilenames, exp_dst_path,
            self.mutex, self.waitCond, rawDataStruct
        )
        if self.rawDataStruct == 2:
            self.worker.basename = self.basename
            self.worker.SizeS = self.SizeS
            self.worker.posNums = self.posNums
            self.worker.chNames = self.chNames
            self.worker.moveOtherFiles = self.moveOtherFiles
            self.worker.copyOtherFiles = self.copyOtherFiles

        self.worker.moveToThread(self.thread)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.taskEnded)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.initPbar.connect(self.setPbarMax)
        self.worker.progressPbar.connect(self.updatePbar)
        self.worker.progress.connect(self.log)
        self.worker.critical.connect(self.workerCritical)
        self.worker.criticalError.connect(self.criticalBioFormats)
        self.worker.confirmMetadata.connect(self.askConfirmMetadata)
        self.worker.filesExisting.connect(self.askReplacePos)
        self.thread.started.connect(self.worker.run)

        self.thread.start()
    
    @exception_handler
    def workerCritical(self, error):
        raise error

    def instructManualStruct(self):
        manual_url = 'https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf'
        txt = (
        f"""
        <p style="font-size:11px;">
        If you would like to add compatibility with your raw microscopy files,<br>
        you can request a new feature <a href=\"{issues_url}">here</a>.<br><br>
        Please label the issue as "enhancement" and provide as many details as
        possible about how your raw microscopy files are structured.<br><br>
        A second option is to create the required data structure manually.
        Please have a look at the instruction on the
        <a href=\"{manual_url}">User manual</a> at the section called
        "Manually create data structure from microscopy file(s)"
        </p>
        """
        )
        msg = QMessageBox(self)
        msg.setWindowTitle('Data structure not available')
        msg.setIcon(msg.Information)
        msg.setText(txt)
        msg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg.setTextFormat(Qt.RichText)
        msg.exec_()

    def addPbar(self):
        self.QPbar = widgets.ProgressBar(self)
        self.QPbar.setValue(0)
        self.mainLayout.insertWidget(3, self.QPbar)

    def updatePbar(self, deltaPbar):
        self.QPbar.setValue(self.QPbar.value()+deltaPbar)

    def setPbarMax(self, max):
        self.QPbar.setMaximum(max)

    def taskEnded(self):
        if self.worker.aborted and not self.worker.isCriticalError:
            msg = QMessageBox(self)
            abort = msg.critical(
               self, 'Conversion task aborted.',
               'Conversion task aborted.',
               msg.Close
            )
            self.close()
        elif not self.worker.aborted:
            msg = QMessageBox(self)
            abort = msg.information(
               self, 'Conversion task ended.',
               'Conversion task ended.\n\n'
               f'Files saved to "{self.worker.exp_dst_path}"',
               msg.Close
            )
            self.close()

    def log(self, text):
        self.logWin.appendPlainText(text)

    def askRawDataStruct(self):
        infoText =  html_utils.paragraph(
            'Select how you have your <b>raw microscopy files arranged</b>'
        )
        win = apps.QDialogCombobox(
            'Raw data structure',
            [
                'Single microscopy file with multiple positions',
                'One or more microscopy files, one for each position',
                'One or more microscopy files, one for each channel',
                'NONE of the above'
            ],
            infoText, CbLabel='', parent=self
        )
        win.exec_()
        return win.selectedItemIdx, win.cancel

    def instructMoveRawFiles(self):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph("""
            Put all of the raw microscopy files from the <b>same experiment</b><br> 
            into an <b>empty folder</b> before closing this dialogue.<br><br>

            Note that there should be <b>no other files</b> in this folder.
        """
        )
        msg.information(
            self, 'Microscopy files location', txt, 
            buttonsTexts=('Cancel', widgets.okPushButton('Done'))
        )
        if msg.cancel:
            return False
        else:
            return True

    def checkFileFormat(self, raw_src_path):
        self.moveOtherFiles = False
        self.copyOtherFiles = False
        ls = natsorted(myutils.listdir(raw_src_path))
        files = [
            filename for filename in ls
            if os.path.isfile(os.path.join(raw_src_path, filename))
        ]
        all_ext = [
            os.path.splitext(filename)[1] for filename in ls
            if os.path.isfile(os.path.join(raw_src_path, filename))
        ]
        counter = Counter(all_ext)
        unique_ext = list(counter.keys())
        is_ext_unique = len(unique_ext) == 1
        most_common_ext, _ = counter.most_common(1)[0]
        if not is_ext_unique:
            if not most_common_ext:
                most_common_ext_msg = '<empty>'
            else:
                most_common_ext_msg = most_common_ext
            
            msg = widgets.myMessageBox(showCentered=False)
            txt = html_utils.paragraph(f"""
                The following folder

                <br><br><code>{raw_src_path}</code><br><br>

                contains files with different file extensions 
                (extensions detected: {unique_ext})<br><br>
                However, the most common extension is 
                <b>{most_common_ext_msg}</b>,
                do you want to proceed with
                loading only files with extension <b>{most_common_ext_msg}</b>?
                <br>
            """)
            _, yesButton, noButton = msg.warning(
                self, 'Multiple extensions detected', txt, 
                buttonsTexts=(
                    'Cancel', 'Yes, load only most common', 
                    'No, load all files'
                )
            )
            if msg.cancel:
                return []
            if msg.clickedButton == yesButton:
                files = [
                    filename for filename in files
                    if os.path.splitext(filename)[1] == most_common_ext
                ]
                otherExt = [
                    ext for ext in unique_ext if ext != most_common_ext]
                files = self.askActionWithOtherFiles(files, otherExt)
            else:
                return files

        if self.rawDataStruct == 0 and len(files) > 1:
            files = self.warnMultipleFiles(files)

        return files

    def askActionWithOtherFiles(self, files, otherExt):
        self.moveOtherFiles = False
        msg = QMessageBox(self)
        msg.setWindowTitle('Action with the other files?')
        txt = (f"""
        <p style="font-size:11px">
            What should I do with the other files (ext: {otherExt})
            in the folder?<br><br>
            <i>NOTE: Only the files with the same basename and position number
            as the raw files will be moved or copied.</i>
        </p>

        """)
        msg.setIcon(msg.Question)
        msg.setText(txt)
        leaveButton = QPushButton(
                'Leave them where they are'
        )
        moveButton = QPushButton(
                'Attempt MOVING to their Position folder'
        )
        copyButton = QPushButton(
                'Attempt COPYING to their Position folder'
        )
        cancelButton = QPushButton(
                'Cancel'
        )
        msg.addButton(leaveButton, msg.YesRole)
        msg.addButton(moveButton, msg.NoRole)
        msg.addButton(copyButton, msg.RejectRole)
        msg.addButton(cancelButton, msg.ApplyRole)
        msg.exec_()
        if msg.clickedButton() == leaveButton:
            self.moveOtherFiles = False
            self.copyOtherFiles = False
            return files
        elif msg.clickedButton() == moveButton:
            self.moveOtherFiles = True
            self.copyOtherFiles = False
            return files
        elif msg.clickedButton() == copyButton:
            self.moveOtherFiles = False
            self.copyOtherFiles = True
            return files
        elif msg.clickedButton() == cancelButton:
            return []


    def warnMultipleFiles(self, files):
        win = apps.QDialogCombobox(
            'Multiple microscopy files detected!', files,
             '<p style="font-size:13px">'
             'You selected "Single microscopy file", '
             'but the <b>folder contains multiple files</b>.<br>'
             '</p>',
             CbLabel='Select which file to load: ', parent=self,
             iconPixmap=QtGui.QPixmap(':warning.svg')
        )
        win.exec_()
        if win.cancel:
            return []
        else:
            files = [win.selectedItemText]

    def attemptSeparateMultiChannel(self, rawFilenames):
        self.chNames = set()
        self.posNums = set()
        stripped_filenames = []
        for file in rawFilenames:
            filename, ext = os.path.splitext(file)
            m_iter = myutils.findalliter(fr'(\d+)_(.+)', filename)
            if len(m_iter) <= 1:
                self.criticalNoFilenamePattern()
                return False
            else:
                m = m_iter[-2]

            try:
                posNum, chName = int(m[0][0]), m[0][1]
                self.chNames.add(chName)
                self.posNums.add(posNum)
                ch_idx = filename.find(f'{posNum}_{chName}')
                stripped_filenames.append(filename[:ch_idx])
            except Exception as e:
                traceback_str = traceback.format_exc()
                self.logger.exception(traceback_str)
                self.criticalNoFilenamePattern(error=traceback.format_exc())
                return False

        basename = myutils.getBasename(stripped_filenames)
        if not basename:
            self.criticalNoFilenamePattern()
            return False

        self.basename = basename

        self.posNums = sorted(list(self.posNums))
        self.chNames = list(self.chNames)
        self.SizeS = len(self.posNums)
        return True


    def criticalNoFilenamePattern(self, error=''):
        txt = (
        """
        <b>Files are named with a non-compatible pattern.</b><br><br>
        In order to automatically generate the required data structure
        from "Multiple files, one for each channel" the filenames must
        be named with a specific pattern:<br><br>
        - basenameN_channelName1, e.g., [ASY015_1_GFP, ASY015_1_mNeon]<br><br>
        where "ASY015" is the basename common to ALL files, "1" is the Position number
        and ["GFP", "mNeon"] are the channel names.<br><br>
        <i>Note that the channel MUST be separated
        from the rest of the name by an underscore "_"</i>
        """
        )
        msg = QMessageBox(self)
        msg.setWindowTitle('Non-compatible pattern')
        msg.setIcon(msg.Critical)
        msg.setText(txt)
        if error:
            msg.setDetailedText(error)
        msg.exec_()

    def criticalBioFormats(self, actionTxt, tracebackFormat, filename):
        msg = QMessageBox(self)
        msg.setIcon(msg.Critical)
        msg.setWindowTitle('Critical error Bio-Formats')
        msg.setDefaultButton(msg.Ok)

        url = 'https://docs.openmicroscopy.org/bio-formats/6.7.0/supported-formats.html'
        seeHere = f'<a href=\"{url}">here</a>'

        _, ext = os.path.splitext(filename)
        txt = (
            f"""
            <p "font-size:11px">
            Error while {actionTxt} with Bio-Formats.<br><br>

            This is most likely because the <b>file format {ext} is not fully supported</b>
            by the Bio-Formats library.<br><br>

            See {seeHere} for details about supported formats.<br><br>

            Try loading file in Fiji and create the data structure manually.<br><br>
            Alternatively, if you are trying to load a video file, you can try
            to open the main GUI and then go to "File --> Open image/video file..."<br><br>
            You were trying to read file: {filename}
            </p>
            """
        )
        msg.setTextFormat(Qt.RichText)
        msg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg.setText(txt)
        msg.setDetailedText(tracebackFormat)
        msg.exec_()
        self.close()

    def askConfirmMetadata(
            self, filename, LensNA, DimensionOrder, SizeT, SizeZ, SizeC, SizeS,
            TimeIncrement, TimeIncrementUnit, PhysicalSizeX, PhysicalSizeY,
            PhysicalSizeZ, PhysicalSizeUnit, chNames, emWavelens, ImageName,
            rawFilePath, sampleImgData
        ):
        if self.rawDataStruct == 2:
            filename = self.basename
        self.metadataDialogIsOpen = True
        if hasattr(self, 'lastDimensionOrder'):
            DimensionOrder = self.lastDimensionOrder
        self.metadataWin = apps.QDialogMetadataXML(
            title=f'Metadata for {filename}', rawFilename=filename,
            LensNA=LensNA, DimensionOrder=DimensionOrder,
            SizeT=SizeT, SizeZ=SizeZ, SizeC=SizeC, SizeS=SizeS,
            TimeIncrement=TimeIncrement, TimeIncrementUnit=TimeIncrementUnit,
            PhysicalSizeX=PhysicalSizeX, PhysicalSizeY=PhysicalSizeY,
            PhysicalSizeZ=PhysicalSizeZ, PhysicalSizeUnit=PhysicalSizeUnit,
            ImageName=ImageName, chNames=chNames, emWavelens=emWavelens,
            parent=self, rawDataStruct=self.rawDataStruct,
            sampleImgData=sampleImgData, rawFilePath=rawFilePath
        )
        self.metadataWin.exec_()
        if not self.metadataWin.cancel:
            self.saveLastSelectedDimensionOrder(self.metadataWin.DimensionOrder)
        self.metadataDialogIsOpen = False
        self.worker.metadataWin = self.metadataWin
        self.waitCond.wakeAll()
    
    def saveLastSelectedDimensionOrder(self, DimensionOrder):
        self.df_settings.at['lastDimensionOrder', 'value'] = DimensionOrder
        self.df_settings.to_csv(settings_csv_path)
        self.lastDimensionOrder = DimensionOrder

    def askReplacePos(self, pos_path):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            f'The following folder <b>already exists</b>.<br><br>'
            f'<code>{pos_path}</code><br><br>'
            'Do you want to <b>overwrite</b> all of its content or '
            '<b>add files</b> to it?'
        )
        cancelButton, overwriteButton, addFilesButton = msg.warning(
           self, 'Replace files?', txt,
           buttonsTexts=('Cancel', 'Overwrite', 'Add files')
        )
        if msg.cancel:
            self.worker.cancel = True
        elif overwriteButton == msg.clickedButton:
            self.worker.overwritePos = True
        elif addFilesButton == msg.clickedButton:
            self.worker.addFiles = True
        self.waitCond.wakeAll()

    def closeEvent(self, event):
        self.logger.info('Closing data structure logger...')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

        if self.buttonToRestore is not None:
            button, color = self.buttonToRestore
            button.setText('0. Attempt "Create data structure" again')
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()

class InitFijiMacro:
    def __init__(self, acdcLauncher):
        self.acdcLauncher = acdcLauncher
        self.logger = self.acdcLauncher.logger
    
    def run(self):
        txt = (f"""    
            In order to run Bio-Formats on your system, Cell-ACDC will use 
            <b>Fiji (ImageJ) from the command line</b>.<br><br>
            The process entails the creation of a macro (.ijm) file and 
            its execution from the command line.<br><br>
            If you prefer to run the macro yourself, you can go through 
            its creation process and cancel its execution later.
        """)
        commands = None
        if not myutils.run_fiji_command():
            try:
                shutil.rmtree(acdc_fiji_path)
            except Exception as err:
                pass
            href = html_utils.href_tag('here', urls.fiji_downloads)
            note_download_txt = (f"""
                Before continuing, Fiji will be <b>automatically downloaded
                now</b>.<br><br>
                If the download fails, please download the zip file from {href} 
                and unzip it in the following location:
            """)
            txt = f'{txt}<br><br>{note_download_txt}'
            commands = (acdc_fiji_path,)
            
        txt = html_utils.paragraph(txt)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(
            self.acdcLauncher, 'Running Fiji in the command line', txt, 
            buttonsTexts=('Cancel', 'Ok'),
            commands=commands
        )
        if msg.cancel:
            self.cancel()
            return
        
        myutils.download_fiji(logger_func=self.logger.info)
        
        win = apps.InitFijiMacroDialog(parent=self.acdcLauncher)
        win.exec_()
        if win.cancel:
            self.cancel()
            return
        
        macro_filepath = fiji_macros.init_macro(*win.init_macro_args)
        macro_command = fiji_macros.command_run_macro(macro_filepath)
        
        txt = html_utils.paragraph("""
            Cell-ACDC will now run the macro in the terminal.<br><br>
            During the process, the <b>GUI will be unresponsive</b>, while 
            progress will be displayed in the terminal.<br><br>
            If you prefer, you can stop the process now and run the command 
            yourself, or even run the macro directly from the Fiji GUI.<br><br>
            Command to run the macro:
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(
            self.acdcLauncher, 'Fiji macro command', txt, 
            buttonsTexts=('Cancel', 'Ok'),
            commands=(macro_filepath)
        )
        if msg.cancel:
            self.cancel()
            return
        
        success = fiji_macros.run_macro(macro_command)
        if success:
            txt = html_utils.paragraph("""
                Macro execution completed successfully. 
                Path to the macro file:
            """)
            msg_func = 'information'
        else:
            href = html_utils.href_tag('GitHub page', urls.issues_url)
            txt = html_utils.paragraph(f"""
                Macro execution completed with errors. More details in the 
                terminal.<br><br> 
                If you cannot solve this, please report the issue on our 
                {href}<br><br>
                Path to the macro file:
            """)
            msg_func = 'information'
        
        msg = widgets.myMessageBox(wrapText=False)
        getattr(msg, msg_func)(
            self.acdcLauncher, 'Macro execution completed', txt
        )
    
    def cancel(self):
        self.logger.info('Running Bio-Formats from Fiji process cancelled.')
        
        
