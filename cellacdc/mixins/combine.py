"""Qt view adapter for the Combine Channels feature."""

from __future__ import annotations

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from qtpy.QtCore import QThread, QTimer, QMutex, QWaitCondition
from natsort import natsorted
import numpy as np

from cellacdc import core, workers, widgets, html_utils, apps, preprocess, myutils, printl


class CombineView:
    """Qt-facing adapter for the Combine Channels feature."""

    LEGACY_METHODS = (
        '_setup_vars_combine',
        'combineDialogSaveCombinedData',
        'combineDialogStepsChanged',
        'updateCombineChannelsPreview',
        'viewCombineChannelDataToggled',
        'setupCombiningChannels',
        'combineDialogClosed',
        '_combineDialogClosed',
        'combineViewAsSegmSetup',
        'combineChannelsActionTriggered',
        'combineEnqueueCurrentImage',
        'combinePreviewToggled',
        'combinePreviewViewAsSegmToggled',
        'combineCurrentImage',
        'combineZStack',
        'combineAllFrames',
        'combineAllPos',
        'stopCombineWorker',
        'combineWorkerCritical',
        'combineWorkerIsQueueEmpty',
        'combineWorkerPreviewDone',
        'combineWorkerAskLoadChannels',
        'combineWorkerDone',
        'combineWorkerClosed',
        'saveCombinedChannelsWorkerFinished',
        'saveCombineWorkerFinished',
    )

    def __init__(self, host):
        object.__setattr__(self, 'host', host)
    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def bind_legacy_methods(self):
        for name in self.LEGACY_METHODS:
            setattr(self.host, name, getattr(self, name))

    def _setup_vars_combine(self):
        self.combineWorker = None
        self.combineDialog = None
        self.combineSegmViewToggle = None
        
    def combineDialogSaveCombinedData(self, dialog):
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

    """Headless state and helpers for combining channel and image arrays."""

    def initialize_combine_image_data(self, pos_data) -> np.ndarray:
        """Initializes pos_data.combine_img_data if not already present."""
        if not hasattr(pos_data, 'combine_img_data'):
            from cellacdc import preprocess
            pos_data.combine_img_data = preprocess.PreprocessedData(
                image_data=np.zeros(pos_data.img_data.shape)
            )
        return pos_data.combine_img_data

    def validate_dimensions(self, ndim: int) -> bool:
        """Asserts that image data dimensions are valid for combining (3D or 4D)."""
        if ndim not in (3, 4):
            raise ValueError('Invalid number of dimensions in img_data.')
        return True

    def group_processed_data_by_pos(
        self,
        processed_data: list[np.ndarray],
        keys: list[tuple[int, int, int]]
    ) -> dict[int, list[tuple[tuple[int, int, int], np.ndarray]]]:
        """Groups raw processed preview output arrays by position index."""
        unique_pos = {key[0] for key in keys}
        per_pos_data = {pos_i: [] for pos_i in unique_pos}
        for key, img in zip(keys, processed_data):
            pos_i, frame_i, z_slice = key
            per_pos_data[pos_i].append((key, img))
        return per_pos_data

    def update_combine_image_data(
        self,
        pos_data,
        pos_i_data: list[tuple[tuple[int, int, int], np.ndarray]]
    ):
        """Updates preprocessed combined image data container frames and z-slices."""
        n_dim_img = pos_data.img_data.ndim
        self.initialize_combine_image_data(pos_data)
        self.validate_dimensions(n_dim_img)

        if n_dim_img == 4:
            for key, img in pos_i_data:
                _, frame_i, z_slice = key
                pos_data.combine_img_data[frame_i][z_slice] = img
        elif n_dim_img == 3:
            for key, img in pos_i_data:
                _, frame_i, _ = key
                pos_data.combine_img_data[frame_i] = img

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
            self.saveCombinedChannelsWorker.deleteLater
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

        self.saveCombinedChannelsWorker.sigDebugShowImg.connect(
            self.preprocessing_view.debugShowImg
        )

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
        else:
            self.setImageImg1()

        if self.viewPreprocDataToggle.isChecked():
            self.viewPreprocDataToggle.toggled.disconnect()  
            self.viewPreprocDataToggle.setChecked(False)
            self.viewPreprocDataToggle.toggled.connect(
                self.preprocessing_view.viewPreprocDataToggled
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
        ordered_channels.extend(segmentations)
        
        self.combineDialog = apps.CombineChannelsSetupDialogGUI(
            ordered_channels,
            isTimelapse=posData.SizeT>1, 
            isZstack=posData.SizeZ>1,
            isMultiPos=len(self.data)>1,
            df_metadata=posData.metadata_df,
            hideOnClosing=True, 
            parent=self
        )
        self.doPreviewPreprocImage = False 
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
        self.preprocessing_view.getChData(requ_ch=selected_channel)

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
        self.preprocessing_view.getChData(requ_ch=selected_channel)

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
        self.preprocessing_view.getChData(requ_ch=selected_channel)

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
            self.preprocessing_view.getChData(
                requ_ch=selected_channel, pos_i=pos_i
            )

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
        per_pos_data = self.group_processed_data_by_pos(processed_data, keys)

        for pos_i, pos_i_data in per_pos_data.items():    
            posData = self.data[pos_i]
            self.update_combine_image_data(posData, pos_i_data)
            
            if not self.combineDialog.saveAsSegm():
                for key, _ in pos_i_data:
                    _, frame_i, z_slice = key
                    self.img1.updateMinMaxValuesCombinedData(
                        self.data, pos_i, frame_i, z_slice
                    )
                if posData.img_data.ndim == 4:
                    self.img1.updateMinMaxValuesCombinedDataProjections(
                        self.data, pos_i, frame_i
                    )
        
        posData = self.data[self.pos_i]
        curr_pos_i, curr_frame_i, curr_z_slice = (
            self.pos_i, self.data[self.pos_i].frame_i, self.z_slice_index()
        )
        if not self.combineDialog.saveAsSegm():
            self.img1.updateMinMaxValuesCombinedData(
                self.data, curr_pos_i, curr_frame_i, curr_z_slice
            )
                
        self.combineViewAsSegmSetup()
        
    def combineWorkerAskLoadChannels(self, requ_channels, pos_i):
        segms_to_load, channels_to_load, current_segm = myutils.separate_fluo_segment_channels(requ_channels)
        if pos_i is None:
            pos_i = list(range(len(self.data)))
        elif not isinstance(pos_i, list):
            pos_i = [pos_i]

        for i in pos_i:
            if channels_to_load:
                self.preprocessing_view.getChData(
                    requ_ch=channels_to_load, pos_i=i
                )
            for segm in segms_to_load:
                self.loadOverlayLabelsData(segm, pos_i=i)
        self.combineWorker.wake_waitCondLoadFluoChannels()
    
    def combineWorkerDone(
            self, 
            processed_data: List[np.ndarray], 
            keys: List[Tuple[int, int, int]]
        ):
        self.status_hover_view.set_status_bar_label(log=False)
        self.combineDialog.appliedFinished()

        per_pos_data = self.group_processed_data_by_pos(processed_data, keys)

        for pos_i, pos_i_data in per_pos_data.items():    
            posData = self.data[pos_i]
            self.update_combine_image_data(posData, pos_i_data)
            
            if not self.combineDialog.saveAsSegm():
                for key, _ in pos_i_data:
                    _, frame_i, z_slice = key
                    self.img1.updateMinMaxValuesCombinedData(
                        self.data, pos_i, frame_i, z_slice
                    )
                if posData.img_data.ndim == 4:
                    self.img1.updateMinMaxValuesCombinedDataProjections(
                        self.data, pos_i, frame_i
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
        
        self.status_hover_view.set_status_bar_label()
        self.logger.info('Combined channels data saved!')
        self.titleLabel.setText('Combined channels data saved!', color='w')

    def saveCombineWorkerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        
        self.status_hover_view.set_status_bar_label()
        self.logger.info('Combined channels saved!')
        self.titleLabel.setText('Combined channels saved!', color='w')