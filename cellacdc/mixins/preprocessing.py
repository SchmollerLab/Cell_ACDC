"""Qt view adapter for image preprocessing workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import numpy as np
from qtpy.QtCore import QMutex, QThread, QWaitCondition

from cellacdc import apps, html_utils, widgets, workers
from cellacdc.plot import imshow

from .session import Session

class Preprocessing(Session):
    """Extracted from guiWin."""

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

    def debugShowImg(self, img):
        imshow(img)

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

    def preprocWorkerClosed(self, worker):
        self.logger.info('Pre-processing worker stopped.')

    def preprocWorkerCritical(self, error):
        self.preprocessDialog.appliedFinished()
        self.workerCritical(error)

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

    def preprocessActionTriggered(self):
        self.preprocessDialog.show()
        self.preprocessDialog.raise_()
        self.preprocessDialog.activateWindow()
        self.preprocessDialog.emitSigPreviewToggled()

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

    def preprocessPreviewToggled(self, checked):
        self.viewPreprocDataToggle.setChecked(checked)
        self.updatePreprocessPreview()

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

    def viewPreprocDataToggled(self, checked):
        self.img1.setUsePreprocessed(checked)
        self.setImageImg1()

        if self.viewCombineChannelDataToggle.isChecked():
            self.viewCombineChannelDataToggle.toggled.disconnect()
            self.viewCombineChannelDataToggle.setChecked(False)
            self.viewCombineChannelDataToggle.toggled.connect(
                self.viewCombineChannelDataToggled
            )
