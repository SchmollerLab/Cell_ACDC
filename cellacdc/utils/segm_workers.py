"""Workers of segmentation-file utilities that cannot live in their utility
module.

CreateConnected3Dsegm, Stack2DsegmTo3Dsegm and FilterObjsFromCoordsTable
have the same names as the utility window classes in
utils/createConnected3Dsegm.py, utils/stack2Dinto3Dsegm.py and
utils/filterObjFromCoordsTable.py. Those modules use them as
segm_workers.<Name>, never through a from-import, which would rebind the
window class name. ApplyTrackInfoWorker is shared by
utils/applyTrackFromTable.py and utils/applyTrackFromTrackMateXML.py.
"""
import re
import os
import json
import numpy as np
import pandas as pd
import skimage.io
import skimage.measure
import skimage.exposure
from qtpy.QtCore import Signal
from cellacdc import load, myutils, segm_re_pattern, io
from cellacdc import core
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base


class CreateConnected3Dsegm(utils_base.BaseWorkerUtil):
    sigAskAppendName = Signal(str, list)
    sigAborted = Signal()

    def __init__(self, mainWin):
        super().__init__(mainWin)

    def criticalSegmIsNot3D(self):
        raise TypeError(
            'Input segmentation masks are not 3D. You can use this utility '
            'only on 3D z-stack data or 4D z-stack over time data.'
        )
    
    @workers_base.worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = f'Select <b>3D segmentation file to connect</b>'
            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.sigAborted.emit()
                return
            
            # Ask appendend name
            self.mutex.lock()
            self.sigAskAppendName.emit(
                self.mainWin.endFilenameSegm, self.mainWin.existingSegmEndNames
            )
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()
            if self.abort:
                self.sigAborted.emit()
                return

            appendedName = self.appendedName
            self.signals.initProgressBar.emit(len(pos_foldernames))
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')
                endFilenameSegm = self.mainWin.endFilenameSegm
                ls = myutils.listdir(images_path)
                file_path = [
                    os.path.join(images_path, f) for f in ls 
                    if f.endswith(f'{endFilenameSegm}.npz')
                ][0]
                
                posData = load.loadData(file_path, '')

                self.signals.sigUpdatePbarDesc.emit(
                    f'Processing {posData.pos_path}')

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_acdc_df=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm
                )
                if posData.segm_data.ndim == 3:
                    posData.segm_data = posData.segm_data[np.newaxis]
                
                self.logger.log('Connecting 3D objects...')
                
                numFrames = len(posData.segm_data)
                self.signals.sigInitInnerPbar.emit(numFrames)
                connectedSegmData = np.zeros_like(posData.segm_data)
                for frame_i, lab in enumerate(posData.segm_data):
                    if lab.ndim != 3:
                        self.criticalSegmIsNot3D()
                        
                    connected_lab = core.connect_3Dlab_zboundaries(lab)
                    connectedSegmData[frame_i] = connected_lab

                    self.signals.sigUpdateInnerPbar.emit(1)

                self.logger.log('Saving connected 3D segmentation file...')
                segmFilename, ext = os.path.splitext(posData.segm_npz_path)
                newSegmFilepath = f'{segmFilename}_{appendedName}.npz'
                connectedSegmData = np.squeeze(connectedSegmData)
                io.savez_compressed(newSegmFilepath, connectedSegmData)
                
                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)

class ApplyTrackInfoWorker(utils_base.BaseWorkerUtil):
    def __init__(
            self, parentWin, endFilenameSegm, trackInfoCsvPath, 
            trackedSegmFilename, trackColsInfo, posPath
        ):
        super().__init__(parentWin)
        self.endFilenameSegm = endFilenameSegm
        self.trackInfoCsvPath = trackInfoCsvPath
        self.trackedSegmFilename = trackedSegmFilename
        self.trackColsInfo = trackColsInfo
        self.posPath = posPath
    
    @workers_base.worker_exception_handler
    def run(self):
        self.logger.log('Loading segmentation file...')  
        self.signals.initProgressBar.emit(0)
        imagesPath = os.path.join(self.posPath, 'Images')
        segmFilename = [
            f for f in myutils.listdir(imagesPath) 
            if f.endswith(f'{self.endFilenameSegm}.npz')
        ][0]
        segmFilePath = os.path.join(imagesPath, segmFilename)
        segmData = np.load(segmFilePath)['arr_0']

        self.logger.log('Loading table containing tracking info...') 
        df = pd.read_csv(self.trackInfoCsvPath)

        frameIndexCol = self.trackColsInfo['frameIndexCol']

        parentIDcol = self.trackColsInfo['parentIDcol']
        pbarMax = len(df[frameIndexCol].unique())
        self.signals.initProgressBar.emit(pbarMax)

        # Apply tracking info
        result = core.apply_tracking_from_table(
            segmData, self.trackColsInfo, df, signal=self.signals.progressBar,
            logger=self.logger.log, pbarMax=pbarMax
        )
        trackedData, trackedIDsMapper, deleteIDsMapper = result

        if self.trackedSegmFilename:
            trackedSegmFilepath = os.path.join(
                imagesPath, self.trackedSegmFilename
            )
        else:
            trackedSegmFilepath = os.path.join(segmFilePath)
        
        self.signals.initProgressBar.emit(0)
        self.logger.log('Saving tracked segmentation file...') 
        io.savez_compressed(trackedSegmFilepath, trackedData)

        
        mapperPath = os.path.splitext(trackedSegmFilepath)[0]
        mapperJsonPath = f'{mapperPath}_deletedIDs_mapper.json'
        mapperJsonName = os.path.basename(mapperJsonPath)
        self.logger.log(f'Saving deleted IDs to {mapperJsonName}...')
        with open(mapperJsonPath, 'w') as file:
            file.write(json.dumps(deleteIDsMapper))

        mapperPath = os.path.splitext(trackedSegmFilepath)[0]
        mapperJsonPath = f'{mapperPath}_replacedIDs_mapper.json'
        mapperJsonName = os.path.basename(mapperJsonPath)
        self.logger.log(f'Saving IDs replacements to {mapperJsonName}...')
        with open(mapperJsonPath, 'w') as file:
            file.write(json.dumps(trackedIDsMapper))

        self.logger.log('Generating acdc_output table...')
        acdc_df = None
        if not self.trackedSegmFilename:
            # Fix existing acdc_df
            acdcEndname = self.endFilenameSegm.replace('_segm', '_acdc_output')
            acdcFilename = [
                f for f in myutils.listdir(imagesPath) 
                if f.endswith(f'{acdcEndname}.csv')
            ]
            if acdcFilename:
                acdcFilePath = os.path.join(imagesPath, acdcFilename[0])
                acdc_df = pd.read_csv(
                    acdcFilePath, index_col=['frame_i', 'Cell_ID']
                )

        if acdc_df is not None:
            acdc_df = core.apply_trackedIDs_mapper_to_acdc_df(
                trackedIDsMapper, deleteIDsMapper, acdc_df
            )
        else:
            acdc_dfs = []
            keys = []
            for frame_i, lab in enumerate(trackedData):            
                rp = skimage.measure.regionprops(lab)
                acdc_df_frame_i = myutils.getBaseAcdcDf(rp)
                acdc_dfs.append(acdc_df_frame_i)
                keys.append(frame_i)
            
            acdc_df = pd.concat(acdc_dfs, keys=keys, names=['frame_i', 'Cell_ID'])
            segmFilename = os.path.basename(trackedSegmFilepath)
            acdcFilename = re.sub(segm_re_pattern, '_acdc_output', segmFilename)
            acdcFilePath = os.path.join(imagesPath, acdcFilename)
        
        self.signals.initProgressBar.emit(pbarMax)
        parentIDcol = self.trackColsInfo['parentIDcol']
        trackIDsCol = self.trackColsInfo['trackIDsCol']
        if parentIDcol != 'None':
            self.logger.log(f'Adding lineage info from "{parentIDcol}" column...')
            acdc_df = core.add_cca_info_from_parentID_col(
                df, acdc_df, frameIndexCol, trackIDsCol, parentIDcol, 
                len(segmData), signal=self.signals.progressBar,
                maskID_colname=self.trackColsInfo['maskIDsCol'], 
                x_colname=self.trackColsInfo['xCentroidCol'], 
                y_colname=self.trackColsInfo['yCentroidCol']
            )     
        
        self.logger.log('Saving acdc_output table...')
        acdc_df.to_csv(acdcFilePath)

        self.signals.finished.emit(self)


class Stack2DsegmTo3Dsegm(utils_base.BaseWorkerUtil):
    sigAskAppendName = Signal(str, list)
    sigAborted = Signal()

    def __init__(self, mainWin, SizeZ):
        super().__init__(mainWin)
        self.SizeZ = SizeZ

    @workers_base.worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = f'Select <b>2D segmentation file to stack</b>'
            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.sigAborted.emit()
                return
            
            # Ask appendend name
            self.mutex.lock()
            self.sigAskAppendName.emit(
                self.mainWin.endFilenameSegm, self.mainWin.existingSegmEndNames
            )
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()
            if self.abort:
                self.sigAborted.emit()
                return

            appendedName = self.appendedName
            self.signals.initProgressBar.emit(len(pos_foldernames))
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')
                endFilenameSegm = self.mainWin.endFilenameSegm
                ls = myutils.listdir(images_path)
                file_path = [
                    os.path.join(images_path, f) for f in ls 
                    if f.endswith(f'{endFilenameSegm}.npz')
                ][0]
                
                posData = load.loadData(file_path, '')

                self.signals.sigUpdatePbarDesc.emit(f'Processing {posData.pos_path}')

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_acdc_df=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm
                )
                if posData.segm_data.ndim == 2:
                    posData.segm_data = posData.segm_data[np.newaxis]
                
                self.logger.log('Stacking 2D into 3D objects...')
                
                numFrames = len(posData.segm_data)
                self.signals.sigInitInnerPbar.emit(numFrames)
                T, Y, X = posData.segm_data.shape
                newShape = (T, self.SizeZ, Y, X)
                segmData2D = np.zeros(newShape, dtype=np.uint32)
                for frame_i, lab in enumerate(posData.segm_data):
                    stacked_lab = core.stack_2Dlab_to_3D(lab, self.SizeZ)
                    segmData2D[frame_i] = stacked_lab

                    self.signals.sigUpdateInnerPbar.emit(1)

                self.logger.log('Saving stacked 3D segmentation file...')
                segmFilename, ext = os.path.splitext(posData.segm_npz_path)
                newSegmFilepath = f'{segmFilename}_{appendedName}.npz'
                segmData2D = np.squeeze(segmData2D)
                io.savez_compressed(newSegmFilepath, segmData2D)
                
                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)


class FilterObjsFromCoordsTable(utils_base.BaseWorkerUtil):
    sigAskAppendName = Signal(str, list)
    sigAborted = Signal()
    sigSetColumnsNames = Signal(object, object, object)

    def __init__(self, mainWin):
        super().__init__(mainWin)

    def emitSetColumnsNames(self, columns, categories, optionalCategories):
        self.mutex.lock()
        self.sigSetColumnsNames.emit(columns, categories, optionalCategories)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort 
    
    def getColumnsCategories(
            self, df_coords, exp_path, pos_foldernames, endFilenameSegm
        ):
        columns = df_coords.columns.to_list()
        categories = ['X coord. column', 'Y coord. column']
        optionalCategories = []
        
        images_path = os.path.join(exp_path, pos_foldernames[0], 'Images')
        metadata_df = load.load_metadata_df(images_path)
        SizeT = float(metadata_df.at['SizeT', 'values'])
        SizeZ = float(metadata_df.at['SizeZ', 'values'])
        
        segmData = load.load_segm_file(
            images_path, end_name_segm_file=endFilenameSegm
        )
        
        if segmData.ndim == 4:
            categories.append('Z coord. column')
            categories.append('Frame index column')
        elif segmData.ndim == 3:
            if SizeZ > 1 and SizeT == 1:
                # 3D z-stack data
                categories.append('Z coord. column')
            else:
                optionalCategories.append('Z coord. column')
                
            if SizeT > 1:
                # 3D time-lapse
                categories.append('Frame index column')
            else:
                optionalCategories.append('Frame index column')
        else:
            optionalCategories.append('Z coord. column')
            optionalCategories.append('Frame index column')
        
        if len(pos_foldernames) > 1:
            categories.append('Position_n')
        else:
            optionalCategories.append('Position_n')
        
        return columns, categories, optionalCategories
        
    def getDfCoords(
            self, df_coords, selectedColumnsPerCategory, pos_foldername, frame_i
        ):
        pos_col = selectedColumnsPerCategory.get('Position_n', 'None')
        frame_i_col = selectedColumnsPerCategory.get(
            'Frame index column', 'None'
        )
        x_col = selectedColumnsPerCategory['X coord. column']
        y_col = selectedColumnsPerCategory['Y coord. column']
        if pos_col != 'None':
            df_coords = df_coords[df_coords[pos_col] == pos_foldername]
        if frame_i_col != 'None':
            df_coords = df_coords[df_coords[frame_i_col] == frame_i]
        
        xy_cols = [x_col, y_col]
        
        df_out = pd.DataFrame(
            index=df_coords.index, 
            data=df_coords[xy_cols].values,
            columns=['x', 'y']
        )
        z_col = selectedColumnsPerCategory.get('Z coord. column', 'None')
        if z_col != 'None':
            df_out['z'] =  df_coords[z_col]
        
        return df_out
    
    @workers_base.worker_exception_handler
    def run(self):
        debugging = False
        expPaths = self.mainWin.expPaths
        tot_exp = len(expPaths)
        self.signals.initProgressBar.emit(0)
        for i, (exp_path, pos_foldernames) in enumerate(expPaths.items()):
            self.errors = {}
            tot_pos = len(pos_foldernames)

            self.mainWin.infoText = f'Select <b>segmentation file to filter</b>'
            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                self.sigAborted.emit()
                return
            endFilenameSegm = self.mainWin.endFilenameSegm
            
            self.logger.log('Asking to select the CSV table file...')
            
            cancel = self.emitSelectFile(
                exp_path, 'Select CSV table file with coordinates to filter',
                'CSV (*.csv)'
            )
            if cancel:
                self.sigAborted.emit()
                return
            
            self.logger.log(
                f'Loading table file `{self.mainWin.selectedFilepath}`..'
            )
            df_coords = pd.read_csv(self.mainWin.selectedFilepath)
            
            columns, categories, optionalCategories = self.getColumnsCategories(
                df_coords, exp_path, pos_foldernames, endFilenameSegm
            )            
            
            abort = self.emitSetColumnsNames(
                columns, categories, optionalCategories
            )
            if abort:
                self.sigAborted.emit()
                return
            
            selectedColumnsPerCategory = self.mainWin.selectedColumnsPerCategory
            
            # Ask appendend name
            self.mutex.lock()
            self.sigAskAppendName.emit(
                self.mainWin.endFilenameSegm, 
                self.mainWin.existingSegmEndNames
            )
            self.waitCond.wait(self.mutex)
            self.mutex.unlock()
            if self.abort:
                self.sigAborted.emit()
                return

            appendedName = self.appendedName
            self.signals.initProgressBar.emit(len(pos_foldernames))
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{tot_exp}, '
                    f'{pos} ({p+1}/{tot_pos})'
                )

                images_path = os.path.join(exp_path, pos, 'Images')
                ls = myutils.listdir(images_path)
                file_path = [
                    os.path.join(images_path, f) for f in ls 
                    if f.endswith(f'{endFilenameSegm}.npz')
                ][0]
                
                posData = load.loadData(file_path, '')

                self.signals.sigUpdatePbarDesc.emit(f'Processing {posData.pos_path}')

                posData.getBasenameAndChNames()
                posData.buildPaths()

                posData.loadOtherFiles(
                    load_segm_data=True,
                    load_acdc_df=True,
                    load_metadata=True,
                    end_filename_segm=endFilenameSegm
                )
                if posData.SizeT == 1:
                    posData.segm_data = posData.segm_data[np.newaxis]
                
                self.logger.log('Filtering objects...')
                
                numFrames = len(posData.segm_data)
                self.signals.sigInitInnerPbar.emit(numFrames)
                filteredSegmData = np.zeros_like(posData.segm_data)
                for frame_i, lab in enumerate(posData.segm_data):
                    df_coords_frame_i = self.getDfCoords(
                        df_coords, selectedColumnsPerCategory, pos, frame_i
                    )
                    if df_coords_frame_i.empty:
                        num_frames_missing = len(posData.segm_data[frame_i:])
                        self.signals.sigUpdateInnerPbar.emit(num_frames_missing)
                        filteredSegmData = filteredSegmData[:frame_i]
                        break
                    
                    filtered_lab = core.filter_segm_objs_from_table_coords(
                        lab, df_coords_frame_i
                    )
                    filteredSegmData[frame_i] = filtered_lab

                    self.signals.sigUpdateInnerPbar.emit(1)

                self.logger.log('Saving filtered segmentation file...')
                segmFilename, ext = os.path.splitext(posData.segm_npz_path)
                newSegmFilepath = f'{segmFilename}_{appendedName}.npz'
                filteredSegmData = np.squeeze(filteredSegmData)
                io.savez_compressed(newSegmFilepath, filteredSegmData)
                
                self.signals.progressBar.emit(1)

        self.signals.finished.emit(self)
