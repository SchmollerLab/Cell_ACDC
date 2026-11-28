"""saveDataWorker, the save worker of the main GUI: segmentation masks,
acdc_output tables (measurements, cell cycle and custom annotations) and
the other per-Position files.

guiWin.saveData builds it with the main window itself; the Compute
measurements utility (utils/compute.py) builds it with its headless main
window. The worker reads guiWin attributes, so its host is always a
guiWin.
"""
import os
import time
import numpy as np
import pandas as pd
import traceback
from qtpy.QtCore import Signal, QObject
from cellacdc import load, printl, io
from cellacdc import cca_df_colnames
from . import base as workers_base


class saveDataWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    sigLog = Signal(str)
    progressBar = Signal(int, int, float)
    critical = Signal(object)
    addMetricsCritical = Signal(str, str)
    regionPropsCritical = Signal(str, str)
    criticalPermissionError = Signal(str)
    metricsPbarProgress = Signal(int, int)
    askZsliceAbsent = Signal(str, object)
    customMetricsCritical = Signal(str, str)
    sigCombinedMetricsMissingColumn = Signal(str, str)
    sigDebug = Signal(object)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.mainWin = mainWin
        self.saveWin = mainWin.saveWin
        self.mutex = mainWin.mutex
        self.waitCond = mainWin.waitCond
        self.customMetricsErrors = {}
        self.addMetricsErrors = {}
        self.regionPropsErrors = {}
        self.abort = False
        self.saveSMBPCcaInfo = False
    
    def checkAbort(self):
        if self.saveWin.aborted:
            self.finished.emit()
            return True
        return False
    
    def saveManualBackgroundData(self, posData, frame_i):
        data_dict = posData.allData_li[frame_i]
        if 'manualBackgroundLab' not in data_dict:
            return
        
        manualBackgrData = data_dict['manualBackgroundLab']
        posData.saveManualBackgroundData(manualBackgrData)
    
    def emitSigPermissionErrorAndSave(
            self, all_frames_acdc_df, acdc_output_csv_path,
            custom_annot_columns
        ):
        err_msg = (
            'The below file is open in another app '
            '(Excel maybe?).\n\n'
            f'{acdc_output_csv_path}\n\n'
            'Close file and then press "Ok".'
        )
        self.mutex.lock()
        self.criticalPermissionError.emit(err_msg)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

        # Save segmentation metadata
        load.save_acdc_df_file(
            all_frames_acdc_df, acdc_output_csv_path, 
            custom_annot_columns=custom_annot_columns, 
            last_cca_frame_i=self.mainWin.save_cca_until_frame_i
        )
        
    def _emitSigDebug(self, stuff_to_debug):
        self.mutex.lock()
        self.sigDebug.emit(stuff_to_debug)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def emitUpdateProgressBar(self):
        t = time.perf_counter()
        exec_time = t - self.time_last_pbar_update
        self.progressBar.emit(1, -1, exec_time)
        self.time_last_pbar_update = t
    
    def addSingleMotherBudPairCcaInfo(
            self, 
            acdc_df_frame_i: pd.DataFrame, 
            posData: load.loadData,
            frame_i: int
        ):
        acdc_df_frame_i['is_single_mother_bud_annotation'] = 0

        if not self.saveSMBPCcaInfo:
            return acdc_df_frame_i
        
        moth_bud_pairs_cca = (
            posData.allData_li[frame_i].get('moth_bud_pairs_cca', None)
        )
        if moth_bud_pairs_cca is None:
            return acdc_df_frame_i

        try:
            idx = moth_bud_pairs_cca.index
            acdc_df_frame_i.loc[idx, cca_df_colnames] = (
                moth_bud_pairs_cca.loc[idx, cca_df_colnames]
            )
            if acdc_df_frame_i['cell_cycle_stage'].isna().any():
                acdc_df_frame_i.loc[idx, 'is_single_mother_bud_annotation'] = 1
        except Exception as err:
            printl(traceback.format_exc())

        return acdc_df_frame_i

    def saveAcdcDf(self, posData: load.loadData, end_i):
        acdc_dfs_li = []
        keys = []
        self.progress.emit(f'Saving annotations for {posData.relPath}...')
        for frame_i, data_dict in enumerate(posData.allData_li[:end_i+1]):
            if self.saveWin.aborted:
                self.finished.emit()
                return

            # Build saved_segm_data
            lab = data_dict['labels']
            if lab is None:
                break
            
            acdc_df = posData.allData_li[frame_i]['acdc_df']
            if acdc_df is None:
                continue
            
            acdc_df = self.addSingleMotherBudPairCcaInfo(
                acdc_df, posData, frame_i
            )

            acdc_dfs_li.append(acdc_df)
            keys.append((frame_i, posData.TimeIncrement*frame_i))
        
        if not acdc_dfs_li:
            return
        
        self.mainWin._measurements_kernel._concat_and_save_acdc_df(
            acdc_dfs_li, keys, posData, self.mainWin.save_metrics,
            saveDataWorker=self, 
            last_cca_frame_i=self.mainWin.save_cca_until_frame_i
        )
    
    def saveSegmData(self, posData: load.loadData, end_i, saved_segm_data):
        self.progress.emit(f'Saving segmentation data for {posData.relPath}...')
        
        
        # extend saved_segm_data if needed
        if posData.SizeT > 1:
            missing_frames_number = end_i + 1 - len(saved_segm_data)
            if missing_frames_number > 0:
                saved_segm_data = np.concatenate(
                    (
                        saved_segm_data,
                        np.zeros(
                            (missing_frames_number, *saved_segm_data.shape[1:]),
                            dtype=saved_segm_data.dtype
                        )
                    ),
                )

        
        for frame_i, data_dict in enumerate(posData.allData_li[:end_i+1]):
            if self.saveWin.aborted:
                self.finished.emit()
                return

            # Build saved_segm_data
            lab = data_dict['labels']
            if lab is None:
                break
            
            posData.lab = lab

            if posData.SizeT > 1:
                saved_segm_data[frame_i] = lab
            else:
                saved_segm_data = lab
                if 'manualBackgroundLab' in data_dict:
                    manualBackgrData = data_dict['manualBackgroundLab']
                    posData.saveManualBackgroundData(manualBackgrData)  
            
        # Save segmentation file
        io.savez_compressed(
            posData.segm_npz_path, np.squeeze(saved_segm_data)
        )
        
        # save information about the segmention
        posData.updateSegmMetadata(all=True)
        posData.saveSegmMetadataIni()
        
        # save rp info about segm
        self.progress.emit(f'Saving additional data for {posData.relPath}...')
        posData.segm_data = saved_segm_data
        # Allow single 2D/3D image
        if posData.SizeT == 1:
            posData.segm_data = posData.segm_data[np.newaxis]
        
        try:
            os.remove(posData.segm_npz_temp_path)
        except Exception as e:
            pass

    @workers_base.worker_exception_handler
    def run(self):
        posToSave = self.mainWin.posToSave
        if posToSave is None:
            numPosToSave = 1
        else:
            numPosToSave = len(posToSave)
        save_metrics = self.mainWin.save_metrics
        if self.isQuickSave:
            save_metrics = False
        self.time_last_pbar_update = time.perf_counter()
        mode = self.mode
        for p, posData in enumerate(self.mainWin.data):
            if self.saveWin.aborted:
                self.finished.emit()
                return
            
            if posToSave is not None:
                if posData.pos_foldername not in posToSave:
                    self.progress.emit(f'Skipping {posData.relPath}')
                    continue
            
            last_tracked_i_path = posData.last_tracked_i_path
            end_i = self.mainWin.save_until_frame_i                
            self.saveSegmData(posData, end_i, posData.segm_data)
            
            posData.saveCustomAnnotationParams()
            current_frame_i = posData.frame_i

            posData.trackedLostCentroidsPath()
            posData.saveTrackedLostCentroids()
            
            
            posData.saveROIInfo()

            if not self.mainWin.isSnapshot:
                last_tracked_i = self.mainWin.last_tracked_i
                if last_tracked_i is None:
                    self.mainWin.saveWin.aborted = True
                    self.finished.emit()
                    return
            elif self.mainWin.isSnapshot:
                last_tracked_i = 0

            if p == 0:
                self.progressBar.emit(0, numPosToSave*(last_tracked_i+1), 0)

            acdc_output_csv_path = posData.acdc_output_csv_path
            delROIs_info_path = posData.delROIs_info_path

            # Add segmented channel data for calc metrics if requested
            add_user_channel_data = True
            for chName in self.mainWin._measurements_kernel.chNamesToSkip:
                skipUserChannel = (
                    posData.filename.endswith(chName)
                    or posData.filename.endswith(f'{chName}_aligned')
                )
                if skipUserChannel:
                    add_user_channel_data = False

            if add_user_channel_data and not self.isQuickSave:
                posData.fluo_data_dict[posData.filename] = posData.img_data
            
            if not self.isQuickSave:
                posData.fluo_bkgrData_dict[posData.filename] = posData.bkgrData
            
            posData.setLoadedChannelNames()
            
            if not self.isQuickSave:
                self.mainWin.initMetricsToSave(posData)
                self.mainWin._measurements_kernel.run(
                    posData=posData, 
                    stop_frame_n=end_i+1,
                    saveDataWorker=self,
                    save_metrics=self.mainWin.save_metrics,
                    last_cca_frame_i=self.mainWin.save_cca_until_frame_i
                )
            else:
                self.saveAcdcDf(posData, end_i)
                
            self.progress.emit(f'Saving {posData.relPath}')

            if not self.do_not_save_og_whitelist:
                og_save_path = os.path.join(
                    posData.images_path,  self.append_name_og_whitelist
                )
                posData.whitelist.saveOGLabs(og_save_path)
            
            if posData.whitelist:
                whitelistIDs_path = posData.segm_npz_path.replace(
                    '.npz', '_whitelistIDs.json'
                )
                new_centroids_path = posData.segm_npz_path.replace(
                    '.npz', '_new_centroids.json'
                )
                posData.whitelist.save(
                    whitelistIDs_path, new_centroids_path=new_centroids_path
                )

            if posData.segmInfo_df is not None:
                try:
                    posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)
                except PermissionError:
                    err_msg = (
                        'The below file is open in another app '
                        '(Excel maybe?).\n\n'
                        f'{posData.segmInfo_df_csv_path}\n\n'
                        'Close file and then press "Ok".'
                    )
                    self.mutex.lock()
                    self.criticalPermissionError.emit(err_msg)
                    self.waitCond.wait(self.mutex)
                    self.mutex.unlock()
                    posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)

            posData.fluo_data_dict.pop(posData.filename, None)

            if not self.isQuickSave:
                posData.fluo_bkgrData_dict.pop(posData.filename)

            if posData.SizeT > 1:
                self.progress.emit('Almost done...')
                self.progressBar.emit(0, 0, 0)

            if self.isQuickSave:
                # Go back to current frame
                posData.frame_i = current_frame_i
                self.mainWin.get_data()
                continue
            
            with open(last_tracked_i_path, 'w+') as txt:
                txt.write(str(end_i))

            # Save combined metrics equations
            posData.saveCombineMetrics()
            self.mainWin.pointsLayerDataToDf(posData)
            posData.saveClickEntryPointsDfs()

            posData.last_tracked_i = last_tracked_i

            # Go back to current frame
            posData.frame_i = current_frame_i
            self.mainWin.get_data()

            if mode == 'Segmentation and Tracking' or mode == 'Viewer':
                self.progress.emit(
                    f'Saved data until frame number {end_i+1}'
                )
            elif mode == 'Cell cycle analysis':
                self.progress.emit(
                    'Saved cell cycle annotations until frame '
                    f'number {self.mainWin.save_cca_until_frame_i+1}'
                )
            # self.progressBar.emit(1)
        if self.mainWin.isSnapshot:
            self.progress.emit(f'Saved all {p+1} Positions!')
        
        self.finished.emit()
