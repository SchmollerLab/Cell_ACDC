from typing import List, Dict, Any, Tuple
from cellacdc import core, workers, widgets, html_utils, apps, preprocess, myutils, printl
from qtpy.QtCore import QThread, QTimer, QMutex, QWaitCondition
from natsort import natsorted
import numpy as np
from typing import Union
# from gui import guiWin

class CombineGuiElements:
    def _setup_vars_combine(self):
        self.combineWorker = None
        self.combineDialog = None
        self.combineSegmViewToggle = None
        
    def combineDialogSaveCombinedData(self, dialog):
        # here check if all data has been processed?
        posData = self.data[self.pos_i]
        
        try:
            posData.combinedChannelsDataArray()
        except TypeError as e:
            if 'Not all frames have been processed.' in str(e):
                msg = widgets.myMessageBox()
                txt = html_utils.paragraph(
                    'Not all frames have been processed.<br>'
                    'Please process all frames before saving.'
                )
                msg.warning(self, 'Process all data before saving', txt)
                return

        helpText = (
            """
            The segm/img file will be saved with a different 
            file name.<br><br>
            Insert a name to append to the end of the new file name. The rest of 
            the name will be the same as the original file base.
            """
        )
        hintText = 'Insert a name for the <b>combined channels</b> file:'
        basename = posData.basename
        if self.combineDialog.saveAsSegm():
            ext = '.npz'
            hintText = hintText.replace('channels', 'segmentation')
            helpText = helpText.replace('channels', 'segmentation')
            basename = f'{basename}segm'
        else:
            ext = '.tif'
            
        win = apps.filenameDialog(
            basename=basename,
            ext=ext,
            hintText=hintText,
            defaultEntry='combined',
            helpText=helpText, 
            allowEmpty=False,
            parent=dialog
        )
        win.exec_()
        if win.cancel:
            return

        appendedText = win.entryText
        if appendedText:
            filename = f'{basename}_{appendedText}{ext}'
        else:
            filename = f'{basename}{ext}'
            
        self.progressWin = apps.QDialogWorkerProgress(
            title='Saving combined channels(s)', 
            parent=self,
            pbarDesc='Saving combined channels(s)'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)
        
        self.statusBarLabel.setText('Saving combined channels...')
        
        self.saveCombinedChannelsWorker = workers.SaveCombinedChannelsWorker(
            self.data, filename,
        )
        
        self.saveCombinedChannelsThread = QThread()
        self.saveCombinedChannelsWorker.moveToThread(self.saveCombinedChannelsThread)
        self.saveCombinedChannelsWorker.signals.finished.connect(
            self.saveCombinedChannelsThread.quit
        )
        self.saveCombinedChannelsWorker.signals.finished.connect(
            self.saveCombinedChannelsWorker.deleteLater
        )
        self.saveCombinedChannelsThread.finished.connect(
            self.saveCombinedChannelsThread.deleteLater
        )
        
        self.saveCombinedChannelsWorker.signals.critical.connect(
            self.workerCritical
        )
        self.saveCombinedChannelsWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.saveCombinedChannelsWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.saveCombinedChannelsWorker.signals.progress.connect(
            self.workerProgress
        )
        self.saveCombinedChannelsWorker.signals.finished.connect(
            self.saveCombinedChannelsWorkerFinished
        )
        
        self.saveCombinedChannelsThread.started.connect(
            self.saveCombinedChannelsWorker.run
        )

        self.saveCombinedChannelsWorker.sigDebugShowImg.connect(self.debugShowImg)

        self.saveCombinedChannelsThread.start()
        
    def combineDialogStepsChanged(self):
        steps, keep_input_data_type, formula = self.combineDialog.steps(return_keepInputDataType=True)
        if steps is None:
            self.logger.warning('Combine channels recipe not initialized yet.')
            return
        
        self.updateCombineChannelsPreview(steps=steps, keep_input_data_type=keep_input_data_type, formula=formula)

    def updateCombineChannelsPreview(self, *args, **kwargs):
        force = kwargs.get('force', False)
        
        if self.combineDialog is None:
            return
        
        if not self.combineDialog.isVisible() and not force:
            return
        
        if not self.combineDialog.previewCheckbox.isChecked() and not force:
            return
        
        if kwargs.get('steps') is None:
            steps, keep_input_data_type, formula = self.combineDialog.steps(return_keepInputDataType=True)
        else:
            steps = kwargs.get('steps')
            keep_input_data_type = kwargs.get('keep_input_data_type')
            formula = kwargs.get('formula')

        if steps is None:
            self.logger.warning('Combine channels recipe not initialized yet.')
            return
        
        txt = 'Combining...'
        self.logger.info(txt)
        self.statusBarLabel.setText(txt)
        
        self.combineEnqueueCurrentImage(steps, keep_input_data_type, formula)
        
    def viewCombineChannelDataToggled(self, checked):
        self.img1.setUseCombined(checked)
        
        if checked:
            self.combineViewAsSegmSetup()
        else: # setimage1 is already called in combineViewAsSegmSetup
            self.setImageImg1()

        if self.viewPreprocDataToggle.isChecked():
            self.viewPreprocDataToggle.toggled.disconnect()  
            self.viewPreprocDataToggle.setChecked(False)
            self.viewPreprocDataToggle.toggled.connect(
                self.viewPreprocDataToggled
            )
            
    def setupCombiningChannels(self):
        posData = self.data[self.pos_i]
        if self.combineDialog is not None:
            self.combineDialog.close()
        
        ordered_channels = [ch for ch in posData.chNames if ch != self.user_ch_name]
        ordered_channels = natsorted(ordered_channels)
        ordered_channels = [self.user_ch_name] + ordered_channels
        

        segmentations = [segm for segm in self.existingSegmEndNames]
        segmentations = natsorted(segmentations)
        segmentations = ['current segm.'] + segmentations
        # also add segm
        ordered_channels.extend(segmentations)
        
        self.combineDialog = apps.CombineChannelsSetupDialogGUI(
            ordered_channels,
            isTimelapse=posData.SizeT>1, 
            isZstack=posData.SizeZ>1,
            isMultiPos=len(self.data)>1,
            df_metadata=posData.metadata_df,
            hideOnClosing=True, 
            # addApplyButton=True,
            parent=self
        )
        self.doPreviewPreprocImage = False #to do
        self.combineDialog.sigApplyImage.connect(
            self.combineCurrentImage
            )
        self.combineDialog.sigApplyZstack.connect(
            self.combineZStack
        )
        self.combineDialog.sigApplyAllFrames.connect(
            self.combineAllFrames
        )
        self.combineDialog.sigApplyAllPos.connect(
            self.combineAllPos
        )
        self.combineDialog.sigPreviewToggled.connect(
            self.combinePreviewToggled
        )
        self.combineDialog.sigSaveAsSegmCheckboxToggled.connect(
            self.combinePreviewViewAsSegmToggled
        )
        self.combineDialog.sigValuesChanged.connect(
            self.combineDialogStepsChanged
        )
        self.combineDialog.sigSavePreprocData.connect(
            self.combineDialogSaveCombinedData
        )
        self.combineDialog.sigClose.connect(
            self.combineDialogClosed
        )

        if self.combineWorker is not None:
            return
        
        self.combineThread = QThread()
        self.combineMutex = QMutex()
        self.combineWaitCond = QWaitCondition()
        
        self.combineWorker = workers.CombineChannelsWorkerGUI(
            self.combineMutex, self.combineWaitCond,
            logger_func=self.logger.info,
            # signals=self.signals # what are the singals for gui???
        )
        
        self.combineWorker.moveToThread(self.combineThread)
        self.combineWorker.signals.finished.connect(self.combineThread.quit)
        self.combineWorker.signals.finished.connect(
            self.combineWorker.deleteLater
        )
        self.combineThread.finished.connect(self.combineWorker.deleteLater)

        self.combineWorker.sigDone.connect(self.combineWorkerDone)
        self.combineWorker.sigIsQueueEmpty.connect(
            self.combineWorkerIsQueueEmpty
        )
        self.combineWorker.sigPreviewDone.connect(self.combineWorkerPreviewDone)
        self.combineWorker.signals.progress.connect(self.workerProgress)
        self.combineWorker.signals.critical.connect(self.workerCritical)
        self.combineWorker.signals.finished.connect(self.combineWorkerClosed)

        self.combineWorker.sigAskLoadChannels.connect(
            self.combineWorkerAskLoadChannels
        )
        
        self.combineThread.started.connect(self.combineWorker.run)
        self.combineThread.start()
        
        self.logger.info('Combine channels worker started.')
    
    def combineDialogClosed(self, window):
        QTimer.singleShot(200, self._combineDialogClosed)
    
    def _combineDialogClosed(self):
        self.combineDialog = None

    def combineViewAsSegmSetup(self):
        if self.combineDialog is None:
            return
        combineViewAsSegm = self.combineDialog.saveAsSegm()
        if not combineViewAsSegm:
            self.img1.setUseCombined(True)
            if self.combineSegmViewToggle.isChecked():
                self.combineSegmViewToggle.setChecked(False)
                self.combineSegmViewToggle.setCheckable(False)
            
        if not self.overlayLabelsButton.isChecked() and combineViewAsSegm:
            self.overlayLabelsButton.blockSignals(True)
            self.overlayLabelsButton.setChecked(True)
            self.overlayLabels_cb(checked=True, selectedLabelsEndnames=['combined segm.'])
            self.overlayLabelsButton.blockSignals(False)
        
        if combineViewAsSegm:
            if not self.combineSegmViewToggle.isChecked():
                self.combineSegmViewToggle.setCheckable(True)
            
            # reset view to update the overlay labels
            self.combineSegmViewToggle.setChecked(False)
            self.combineSegmViewToggle.setChecked(True)

            self.img1.setUseCombined(False)
        self.setImageImg1()

    def combineChannelsActionTriggered(self):
        if self.zProjComboBox is not None:
            curr_proj = self.zProjComboBox.currentText()
            if curr_proj != 'single z-slice':
                self.zProjComboBox.setCurrentText('single z-slice')
          
        if self.switchPlaneCombobox is not None:
            depthAxes = self.switchPlaneCombobox.depthAxes()
            if depthAxes != 'z':
                self.switchPlaneCombobox.setCurrentText('xy')
        
        if self.combineDialog is None:
            self.setupCombiningChannels()
        self.combineDialog.show()
        self.combineDialog.raise_()
        self.combineDialog.activateWindow()
        self.combineDialog.emitSigPreviewToggled()
        
class CombineGUIWorker:
    def combineEnqueueCurrentImage(self, steps, keep_input_data_type, formula):
        posData = self.data[self.pos_i]

        if posData.SizeZ > 1:
            z_slice = self.z_slice_index()
        else:
            z_slice = 0
        
        key = (self.pos_i, posData.frame_i, z_slice)
        self.combineWorker.enqueue(
            self.data,
            steps, 
            key,
            keep_input_data_type,
            output_as_segm=self.combineDialog.saveAsSegm(),
            formula=formula,
        )
        
    def combinePreviewToggled(self, checked):
        self.viewCombineChannelDataToggle.setChecked(checked)
        self.updateCombineChannelsPreview()
        
    def combinePreviewViewAsSegmToggled(self, checked):
        self.updateCombineChannelsPreview()
        self.combineViewAsSegmSetup()
            
    def combineCurrentImage(
            self, 
            steps: List[Dict[str, Any]]=None,
            keep_input_data_type:bool=None,
            formula: str=None,
        ):

        if steps and keep_input_data_type is None:
            raise ValueError('keep_input_data_type must be set if steps is set')
        
        if steps is None:
            steps, keep_input_data_type, formula = self.combineDialog.steps(
                return_keepInputDataType=True
            )
        txt = 'Combining current image...'
        self.logger.info(txt)
        self.statusBarLabel.setText(txt)
        
        selected_channel = core.get_selected_channels(steps)
        self.getChData(requ_ch=selected_channel)

        z_slice = self.zSliceScrollBar.sliderPosition()
        pos_i = self.pos_i

        key = (pos_i, self.data[pos_i].frame_i, z_slice)

        self.combineWorker.setupJob(
            self.data, 
            steps, 
            keep_input_data_type,
            key,
            output_as_segm=self.combineDialog.saveAsSegm(),
            formula=formula,
        )
        
        self.combineWorker.wakeUp()
    
    def combineZStack(
            self, 
            steps: List[Dict[str, Any]]=None, 
            keep_input_data_type:bool=None,
            formula: str=None,
        ):
        if self.combineDialog is not None:
            keep_input_data_type = (
                self.combineDialog.keepInputDataTypeToggle.isChecked()
            )
        
        if steps and keep_input_data_type is None:
            raise ValueError('keep_input_data_type must be set if steps is set')
        
        if steps is None:
            steps, keep_input_data_type, formula = self.combineDialog.steps(
                return_keepInputDataType=True
            )
        txt = 'Combining z-stack...'
        self.statusBarLabel.setText(txt)
        self.logger.info(txt)
        
        selected_channel = core.get_selected_channels(steps)
        self.getChData(requ_ch=selected_channel)

        posData = self.data[self.pos_i]
        key = (self.pos_i, posData.frame_i, None)
        self.combineWorker.setupJob(
            self.data, 
            steps, 
            keep_input_data_type,
            key,
            output_as_segm=self.combineDialog.saveAsSegm(),
            formula=formula,
        )

        self.combineWorker.wakeUp()
    
    def combineAllFrames(self, 
                         steps: List[Dict[str, Any]]=None,
                         keep_input_data_type:bool=None,
                         formula: str=None,
                         ):
        if steps and not keep_input_data_type:
            raise ValueError('keep_input_data_type must be set if steps is set')
        
        if steps is None:
            steps, keep_input_data_type, formula = self.combineDialog.steps(return_keepInputDataType=True)
        txt = 'Combining all frames...'
        self.logger.info(txt)
        self.statusBarLabel.setText(txt)
        
        selected_channel = core.get_selected_channels(steps)
        self.getChData(requ_ch=selected_channel)

        key = (self.pos_i, None, None)
        self.combineWorker.setupJob(
            self.data, 
            steps, 
            keep_input_data_type,
            key,
            output_as_segm=self.combineDialog.saveAsSegm(),
            formula=formula,
        )

        self.combineWorker.wakeUp()
    
    def combineAllPos(self, 
                      steps: List[Dict[str, Any]]=None,
                      keep_input_data_type:bool=None,
                      formula: str=None,
                      ):
        if steps and not keep_input_data_type:
            raise ValueError('keep_input_data_type must be set if steps is set')
        
        if steps is None:
            steps, keep_input_data_type, formula = self.combineDialog.steps(return_keepInputDataType=True)
        txt = 'Combining all Positions...'
        self.logger.info(txt)
        self.statusBarLabel.setText(txt)
        
        selected_channel = core.get_selected_channels(steps)
        
        for pos_i in range(len(self.data)):
            self.getChData(requ_ch=selected_channel, pos_i=pos_i)


        key = (None, None, None)
        self.combineWorker.setupJob(
            self.data, 
            steps, 
            keep_input_data_type,
            key,
            output_as_segm=self.combineDialog.saveAsSegm(),
            formula=formula,
        )

        self.combineWorker.wakeUp()
        
    def stopCombineWorker(self):
        self.logger.info('Closing combine worker...')
        try:
            self.combineWorker.stop()
        except Exception as err:
            pass
        
    def combineWorkerCritical(self, error):
        self.combineDialog.appliedFinished()
        self.workerCritical(error)

    def combineWorkerIsQueueEmpty(self, isEmpty: bool):
        if isEmpty:
            self.combineDialog.appliedFinished()
        else:
            self.combineDialog.setDisabled(True)
            self.combineDialog.infoLabel.setText(
                'Computing preview...<br>'
                '<i>(Feel free to use Cell-ACDC while waiting)</i>'
            )

    def combineWorkerPreviewDone(
            self, 
            processed_data: List[np.ndarray], 
            keys: List[Tuple[int, int, int]]
        ):
        unique_pos = {key[0] for key in keys}
        per_pos_data = {pos_i: [] for pos_i in unique_pos}

        for key, img in zip(keys, processed_data):
            pos_i, frame_i, z_slice = key
            per_pos_data[pos_i].append((key, img))

        for pos_i in unique_pos:    
            posData = self.data[pos_i]
            if not hasattr(posData, 'combine_img_data'):
                posData.combine_img_data = preprocess.PreprocessedData(
                    image_data=np.zeros(posData.img_data.shape)
                )

            n_dim_img = posData.img_data.ndim

            if n_dim_img == 4:
                for key, processed_data in per_pos_data[pos_i]:
                    pos_i, frame_i, z_slice = key
                    posData.combine_img_data[frame_i][z_slice] = processed_data
                    if not self.combineDialog.saveAsSegm():
                        self.img1.updateMinMaxValuesCombinedData(
                            self.data, pos_i, frame_i, z_slice
                        )
                self.img1.updateMinMaxValuesCombinedDataProjections(
                    self.data, pos_i, frame_i
                )
            elif n_dim_img == 3:
                for key, processed_data in per_pos_data[pos_i]:
                    pos_i, frame_i, z_slice = key
                    posData.combine_img_data[frame_i] = processed_data
                    if not self.combineDialog.saveAsSegm():
                        self.img1.updateMinMaxValuesCombinedData(
                            self.data, pos_i, frame_i, z_slice
                        )
            else:
                raise ValueError('Invalid number of dimensions in img_data.')
        
        posData = self.data[self.pos_i]
        curr_pos_i, curr_frame_i, curr_z_slice = (
            self.pos_i,self.data[self.pos_i].frame_i, self.z_slice_index()
        )
        if not self.combineDialog.saveAsSegm():
            self.img1.updateMinMaxValuesCombinedData(
                self.data, curr_pos_i, curr_frame_i, curr_z_slice
            )
                
        self.combineViewAsSegmSetup()
        
    def combineWorkerAskLoadChannels(self, requ_channels, pos_i):
        # spit channels and segm to load
        segms_to_load, channels_to_load, current_segm = myutils.separate_fluo_segment_channels(requ_channels)
        if pos_i is None:
            pos_i = list(range(len(self.data)))
        elif not isinstance(pos_i, list):
            pos_i = [pos_i]

        for i in pos_i:
            if channels_to_load:
                self.getChData(requ_ch=channels_to_load, pos_i=i)
            for segm in segms_to_load:
                self.loadOverlayLabelsData(segm, pos_i=i)
        self.combineWorker.wake_waitCondLoadFluoChannels()
    
    def combineWorkerDone(
            self, 
            processed_data: List[np.ndarray], 
            keys: List[Tuple[int, int, int]]
        ):
        self.setStatusBarLabel(log=False)
        self.combineDialog.appliedFinished()

        unique_pos = {key[0] for key in keys}
        per_pos_data = {pos_i: [] for pos_i in unique_pos}

        for key, img in zip(keys, processed_data):
            pos_i, frame_i, z_slice = key
            per_pos_data[pos_i].append((key, img))

        for pos_i in unique_pos:    
            posData = self.data[pos_i]
            if not hasattr(posData, 'combine_img_data'):
                posData.combine_img_data = preprocess.PreprocessedData(
                    image_data=np.zeros(posData.img_data.shape)
                )

            n_dim_img = posData.img_data.ndim

            if n_dim_img == 4:
                for key, processed_data in per_pos_data[pos_i]:
                    pos_i, frame_i, z_slice = key
                    posData.combine_img_data[frame_i][z_slice] = processed_data
                    if not self.combineDialog.saveAsSegm():
                        self.img1.updateMinMaxValuesCombinedData(
                                self.data, pos_i, frame_i, z_slice
                            )
                if not self.combineDialog.saveAsSegm():        
                    self.img1.updateMinMaxValuesCombinedDataProjections(
                        self.data, pos_i, frame_i
                    )
            else:
                for key, processed_data in per_pos_data[pos_i]:
                    pos_i, frame_i, z_slice = key
                    posData.combine_img_data[frame_i] = processed_data
                    if not self.combineDialog.saveAsSegm():
                        self.img1.updateMinMaxValuesCombinedData(
                            self.data, pos_i, frame_i, z_slice
                        )
                
            if not self.viewCombineChannelDataToggle.isChecked():
                self.viewCombineChannelDataToggle.setChecked(True)
            else:
                self.setImageImg1()

    def combineWorkerClosed(self, worker):
        self.logger.info('Combine worker stopped.')
        
    def saveCombinedChannelsWorkerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        
        self.setStatusBarLabel()
        self.logger.info('Combined channels data saved!')
        self.titleLabel.setText('Combined channels data saved!', color='w')

    def saveCombineWorkerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        
        self.setStatusBarLabel()
        self.logger.info('Combined channels saved!')
        self.titleLabel.setText('Combined channels saved!', color='w')


class PreprocessingGuiElements:
    """Preprocessing recipes in the main GUI: preview, apply to the frame,
    z-stack, all frames or all positions, and save.
    """

    def preprocWorkerIsQueueEmpty(self, isEmpty: bool):
        if isEmpty:
            self.preprocessDialog.appliedFinished()
        else:
            self.preprocessDialog.setDisabled(True)
            self.preprocessDialog.infoLabel.setText(
                'Computing preview...<br>'
                '<i>(Feel free to use Cell-ACDC while waiting)</i>'
            )

    def preprocWorkerPreviewDone(
            self, processed_data: np.ndarray, 
            key: Tuple[int, int, Union[int, str]]
        ):
        pos_i, frame_i, z_slice = key
        posData = self.data[pos_i]
        if not hasattr(posData, 'preproc_img_data'):
            posData.preproc_img_data = preprocess.PreprocessedData(
                image_data=np.zeros(posData.img_data.shape)
            )
        
        posData.preproc_img_data[frame_i][z_slice] = processed_data
        self.img1.updateMinMaxValuesPreprocessedData(
            self.data, pos_i, frame_i, z_slice
        )
        
        self.setImageImg1()

    def preprocWorkerDone(
            self, 
            processed_data: np.ndarray, 
            how: str, 
        ):
        self.setStatusBarLabel(log=False)
        self.preprocessDialog.appliedFinished()
            
        posData = self.data[self.pos_i]
        if not hasattr(posData, 'preproc_img_data'):
            posData.preproc_img_data = preprocess.PreprocessedData()

        if how == 'current_image':
            if posData.SizeZ > 1:
                z_slice = self.z_slice_index()
                posData.preproc_img_data[posData.frame_i][z_slice] = (
                    processed_data
                )
            else:
                posData.preproc_img_data[posData.frame_i] = processed_data
                z_slice = 0
            self.img1.updateMinMaxValuesPreprocessedData(
                self.data, self.pos_i, posData.frame_i, z_slice
            )
        elif how == 'z_stack':
            for z_slice, processed_img in enumerate(processed_data):
                posData.preproc_img_data[posData.frame_i][z_slice] = (
                    processed_img
                )
                self.img1.updateMinMaxValuesPreprocessedData(
                    self.data, self.pos_i, posData.frame_i, z_slice
                )
                self.img1.updateMinMaxValuesPreprocessedProjections(
                    self.data, self.pos_i, posData.frame_i
                )
        elif how == 'all_frames':
            for frame_i, processed_frame in enumerate(processed_data):
                if processed_frame.ndim == 2:
                    processed_frame = (processed_frame,)
                    
                for z_slice, processed_img in enumerate(processed_frame):
                    posData.preproc_img_data[frame_i][z_slice] = (
                        processed_img
                    )
                    self.img1.updateMinMaxValuesPreprocessedData(
                        self.data, self.pos_i, frame_i, z_slice
                    )
                self.img1.updateMinMaxValuesPreprocessedProjections(
                    self.data, self.pos_i, frame_i
                )
        elif how == 'all_pos':
            for pos_i, processed_pos_data in enumerate(processed_data):                    
                if processed_pos_data.ndim == 2:
                    processed_pos_data = (processed_pos_data,)

                posData = self.data[pos_i]
                if not hasattr(posData, 'preproc_img_data'):
                    posData.preproc_img_data = preprocess.PreprocessedData()
                for z_slice, processed_img in enumerate(processed_pos_data):
                    posData.preproc_img_data[0][z_slice] = (
                        processed_img
                    )
                    self.img1.updateMinMaxValuesPreprocessedData(
                        self.data, pos_i, 0, z_slice
                    )
                
                if posData.SizeZ > 1:
                    self.img1.updateMinMaxValuesPreprocessedProjections(
                        self.data, pos_i, frame_i
                    )
            
        if not self.viewPreprocDataToggle.isChecked():
            self.viewPreprocDataToggle.setChecked(True)
        else:
            self.setImageImg1()

    def preprocWorkerClosed(self, worker):
        self.logger.info('Pre-processing worker stopped.')

    def savePreprocWorkerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        
        self.setStatusBarLabel()
        self.logger.info('Pre-processed data saved!')
        self.titleLabel.setText('Pre-processed data saved!', color='w')

    def askGet2Dor3Dimage(self):
        txt = html_utils.paragraph("""
            Do you want to test the denoising on the visualized 2D image or 
            on the entire 3D z-stack?                           
        """)
        msg = widgets.myMessageBox(wrapText=False)
        _, use3Dbutton, use2Dbutton = msg.question(
            self, '3D denoising?', txt, 
            buttonsTexts=('Cancel', 'Denoise 3D z-stack', 'Denoise 2D image')
        )
        if msg.cancel:
            return 
        
        if msg.clickedButton == use3Dbutton:
            posData = self.data[self.pos_i]
            zslice = self.zSliceScrollBar.sliderPosition()
            return posData.img_data[posData.frame_i, zslice]
        else:
            return self.getDisplayedImg1()

    def preprocessActionTriggered(self):
        self.preprocessDialog.show()
        self.preprocessDialog.raise_()
        self.preprocessDialog.activateWindow()
        self.preprocessDialog.emitSigPreviewToggled()

    def preprocessDialogRecipeChanged(self, recipe):# why does this need the recepie as an arg
        recipe = self.preprocessDialog.recipe()
        if recipe is None:
            self.logger.warning('Pre-processing recipe not initialized yet.')
            return
        
        self.updatePreprocessPreview(recipe=recipe)

    def preprocessDialogSavePreprocessedData(self, dialog):
        posData = self.data[self.pos_i]
        
        try:
            posData.preprocessedDataArray()
        except TypeError as e:
            if 'Not all frames have been processed.' in str(e):
                msg = widgets.myMessageBox()
                txt = html_utils.paragraph(
                    'Not all frames have been processed.<br>'
                    'Please process all frames before saving.'
                )
                msg.warning(self, 'Process all data before saving', txt)
                return


        helpText = (
            """
            The preprocessed image file will be saved with a different 
            file name.<br><br>
            Insert a name to append to the end of the new file name. The rest of 
            the name will be the same as the original file.
            """
        )
        
        
        win = apps.filenameDialog(
            basename=f'{posData.basename}{self.user_ch_name}',
            ext=".tif",
            hintText='Insert a name for the <b>preprocessed image</b> file:',
            defaultEntry='preprocessed',
            helpText=helpText, 
            allowEmpty=False,
            parent=dialog
        )
        win.exec_()
        if win.cancel:
            return

        appendedText = win.entryText
        
        self.progressWin = apps.QDialogWorkerProgress(
            title='Saving pre-processed image(s)', 
            parent=self,
            pbarDesc='Saving pre-processed image(s)'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)
        
        self.statusBarLabel.setText('Saving pre-processed data...')
        
        self.savePreprocWorker = workers.SaveProcessedDataWorker(
            self.data, appendedText, ext=".tif"
        )
        
        self.savePreprocThread = QThread()
        self.savePreprocWorker.moveToThread(self.savePreprocThread)
        self.savePreprocWorker.signals.finished.connect(
            self.savePreprocThread.quit
        )
        self.savePreprocWorker.signals.finished.connect(
            self.savePreprocWorker.deleteLater
        )
        self.savePreprocThread.finished.connect(
            self.savePreprocThread.deleteLater
        )
        
        self.savePreprocWorker.signals.critical.connect(
            self.workerCritical
        )
        self.savePreprocWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.savePreprocWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.savePreprocWorker.signals.progress.connect(
            self.workerProgress
        )
        self.savePreprocWorker.signals.finished.connect(
            self.savePreprocWorkerFinished
        )
        
        self.savePreprocThread.started.connect(
            self.savePreprocWorker.run
        )
        self.savePreprocThread.start()


    def preprocessEnqueueCurrentImage(self, recipe):
        posData = self.data[self.pos_i]
        func = core.preprocess_image_from_recipe
        image_data = self.getImage(raw=True)
        if posData.SizeZ > 1:
            z_slice = self.z_slice_index()
        else:
            z_slice = 0
        
        recipe = core.validate_multidimensional_recipe(recipe)
        
        key = (self.pos_i, posData.frame_i, z_slice)
        self.preprocWorker.enqueue(
            func, 
            image_data, 
            recipe,
            key
        )

    def getChData(self, requ_ch=None, pos_i=None):
        if not pos_i:
            pos_i = self.pos_i

        posData = self.data[pos_i]

        if not requ_ch:
            requ_ch = set(self.ch_names)
        else:
            requ_ch = set(requ_ch)

        posData.setLoadedChannelNames()

        loaded_channels = set(posData.loadedChNames)
        missing_channels = requ_ch - loaded_channels

        self.loadFluo_cb(fluo_channels=missing_channels)

    def updatePreprocessPreview(self, *args, **kwargs):
        force = kwargs.get('force', False)
        
        if not self.preprocessDialog.isVisible() and not force:
            return
        
        if not self.preprocessDialog.previewCheckbox.isChecked() and not force:
            return
        
        if kwargs.get('recipe') is None:
            recipe = self.preprocessDialog.recipe()
        else:
            recipe = kwargs.get('recipe')

        if recipe is None:
            self.logger.warning('Pre-processing recipe not initialized yet.')
            return
        
        txt = 'Pre-processing current image...'
        self.logger.info(txt)
        self.statusBarLabel.setText(txt)
        
        self.preprocessEnqueueCurrentImage(recipe)

    def preprocessPreviewToggled(self, checked):
        self.viewPreprocDataToggle.setChecked(checked)
        self.updatePreprocessPreview()



    def preprocessCurrentImage(self, recipe: List[Dict[str, Any]], *args):
        txt = 'Pre-processing current image...'
        self.logger.info(txt)
        self.statusBarLabel.setText(txt)
        
        func = core.preprocess_image_from_recipe
        recipe = core.validate_multidimensional_recipe(recipe)
        
        image_data = self.getImage(raw=True)
        self.preprocWorker.setupJob(
            func, 
            image_data, 
            recipe, 
            'current_image'
        )
        
        self.preprocWorker.wakeUp()

    def preprocessZStack(self, recipe: List[Dict[str, Any]], *args):
        txt = 'Pre-processing z-stack...'
        self.statusBarLabel.setText(txt)
        self.logger.info(txt)
        
        posData = self.data[self.pos_i]
        func = core.preprocess_zstack_from_recipe
        recipe = core.validate_multidimensional_recipe(
            recipe, apply_to_all_frames=False
        )
        image_data = posData.img_data[posData.frame_i]
        self.preprocWorker.setupJob(
            func, 
            image_data, 
            recipe, 
            'z_stack'
        )
        
        self.preprocWorker.wakeUp()

    def preprocessAllFrames(self, recipe: List[Dict[str, Any]]):
        txt = 'Pre-processing all frames...'
        self.logger.info(txt)
        self.statusBarLabel.setText(txt)
        
        posData = self.data[self.pos_i]
        func = core.preprocess_video_from_recipe
        image_data = posData.img_data
        self.preprocWorker.setupJob(
            func, 
            image_data, 
            recipe, 
            'all_frames'
        )
        self.preprocWorker.wakeUp()

    def preprocessAllPos(self, recipe: List[Dict[str, Any]]):
        txt = 'Pre-processing all Positions...'
        self.logger.info(txt)
        self.statusBarLabel.setText(txt)
        
        func = core.preprocess_multi_pos_from_recipe
        recipe = core.validate_multidimensional_recipe(
            recipe, apply_to_all_frames=False
        )
        image_data = [posData.img_data[0] for posData in self.data]
        self.preprocWorker.setupJob(
            func, 
            image_data, 
            recipe, 
            'all_pos'
        )
        
        self.preprocWorker.wakeUp()

    def setupPreprocessing(self):
        posData = self.data[self.pos_i]
        if self.preprocessDialog is not None:
            self.preprocessDialog.close()
        
        self.preprocessDialog = apps.PreProcessRecipeDialog(
            isTimelapse=posData.SizeT>1, 
            isZstack=posData.SizeZ>1,
            isMultiPos=len(self.data)>1,
            df_metadata=posData.metadata_df,
            hideOnClosing=True, 
            addApplyButton=True,
            parent=self
        )
        self.doPreviewPreprocImage = False
        self.preprocessDialog.sigApplyImage.connect(
            self.preprocessCurrentImage
        )
        self.preprocessDialog.sigApplyZstack.connect(
            self.preprocessZStack
        )
        self.preprocessDialog.sigApplyAllFrames.connect(
            self.preprocessAllFrames
        )
        self.preprocessDialog.sigApplyAllPos.connect(
            self.preprocessAllPos
        )
        self.preprocessDialog.sigPreviewToggled.connect(
            self.preprocessPreviewToggled
        )
        self.preprocessDialog.sigValuesChanged.connect(
            self.preprocessDialogRecipeChanged
        )
        self.preprocessDialog.sigSavePreprocData.connect(
            self.preprocessDialogSavePreprocessedData
        )
        
        if self.preprocWorker is not None:
            return
        
        self.preprocThread = QThread()
        self.preprocMutex = QMutex()
        self.preprocWaitCond = QWaitCondition()
        
        self.preprocWorker = workers.CustomPreprocessWorkerGUI(
            self.preprocMutex, self.preprocWaitCond
        )
        
        self.preprocWorker.moveToThread(self.preprocThread)
        self.preprocWorker.signals.finished.connect(self.preprocThread.quit)
        self.preprocWorker.signals.finished.connect(
            self.preprocWorker.deleteLater
        )
        self.preprocThread.finished.connect(self.preprocThread.deleteLater)

        self.preprocWorker.sigDone.connect(self.preprocWorkerDone)
        self.preprocWorker.sigIsQueueEmpty.connect(
            self.preprocWorkerIsQueueEmpty
        )
        self.preprocWorker.sigPreviewDone.connect(self.preprocWorkerPreviewDone)
        self.preprocWorker.signals.progress.connect(self.workerProgress)
        self.preprocWorker.signals.critical.connect(self.workerCritical)
        self.preprocWorker.signals.finished.connect(self.preprocWorkerClosed)
        
        self.preprocThread.started.connect(self.preprocWorker.run)
        self.preprocThread.start()
        
        self.logger.info('Pre-processing worker started.')

    def preprocWorkerCritical(self, error):
        self.preprocessDialog.appliedFinished()
        self.workerCritical(error)

    def viewPreprocDataToggled(self, checked):
        self.img1.setUsePreprocessed(checked)
        self.setImageImg1()

        if self.viewCombineChannelDataToggle.isChecked():
            self.viewCombineChannelDataToggle.toggled.disconnect()
            self.viewCombineChannelDataToggle.setChecked(False)
            self.viewCombineChannelDataToggle.toggled.connect(
                self.viewCombineChannelDataToggled
            )

    def stopPreprocWorker(self):
        self.logger.info('Closing pre-processing worker...')
        try:
            self.preprocWorker.stop()
        except Exception as err:
            pass
