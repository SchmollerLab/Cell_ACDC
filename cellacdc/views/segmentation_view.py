"""Qt view adapter for segmentation workflows."""

from __future__ import annotations

import os

import numpy as np
from qtpy.QtCore import QMutex, QThread, QTimer, QWaitCondition, Qt
from qtpy.QtWidgets import QAction

from cellacdc import (
    apps, exception_handler, html_utils, prompts, printl, widgets, workers,
)
from cellacdc.plot import imshow
from cellacdc.viewmodels.segmentation_viewmodel import SegmentationViewModel


class SegmentationView:
    """Qt-facing segmentation workflow adapter."""

    LEGACY_METHODS = (
        'computeSegm',
        'autoSegm_cb',
        'postProcessSegm',
        'postProcessSegmApplyToAllFutureFrames',
        'postProcessSegmEditingFinished',
        'postProcessSegmWorkerFinished',
        'postProcessSegmWinClosed',
        'postProcessSegmValueChanged',
        'resetCursor',
        'segmFrameCallback',
        'showInstructionsCustomModel',
        'reinitStoredSegmModels',
        'segmVideoCallback',
        'segmentToolActionTriggered',
        'initSegmModelParams',
        'repeatSegm',
        'debugSegmWorker',
        'selectZtoolZvalueChanged',
        'repeatSegmVideo',
        'segmVideoWorkerFinished',
        'segmWorkerFinished',
        'postProcessing',
        'checkIfAutoSegm',
        'init_segmInfo_df',
    )

    def __init__(self, host, view_model: SegmentationViewModel):
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'view_model', view_model)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host', 'view_model'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def bind_legacy_methods(self):
        for name in self.LEGACY_METHODS:
            setattr(self.host, name, getattr(self, name))

    def computeSegm(self, force=False):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        should_compute = self.view_model.should_compute_segmentation(
            mode=mode,
            has_labels=np.any(posData.lab),
            force=force,
            auto_enabled=self.autoSegmAction.isChecked(),
        )
        if not should_compute:
            return

        self.repeatSegm(model_name=self.segmModelName)

    def autoSegm_cb(self, checked):
        if checked:
            self.askSegmParam = True
            # Ask which model
            models = self.view_model.segmentation_models()
            win = widgets.QDialogListbox(
                'Select model',
                'Select model to use for segmentation: ',
                models,
                multiSelection=False,
                parent=self.host
            )
            win.exec_()
            if win.cancel:
                return
            model_name = win.selectedItemsText[0]
            self.segmModelName = model_name
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            self.updateAllImages()
            self.computeSegm()
            self.askSegmParam = False
        else:
            self.segmModelName = None

    def postProcessSegm(self, checked):
        if self.isSegm3D:
            SizeZ = max([posData.SizeZ for posData in self.data])
        else:
            SizeZ = None
        if checked:
            posData = self.data[self.pos_i]
            self.postProcessSegmWin = apps.PostProcessSegmDialog(
                posData, mainWin=self.host
            )
            self.postProcessSegmWin.sigClosed.connect(
                self.postProcessSegmWinClosed
            )
            self.postProcessSegmWin.sigValueChanged.connect(
                self.postProcessSegmValueChanged
            )
            self.postProcessSegmWin.sigEditingFinished.connect(
                self.postProcessSegmEditingFinished
            )
            self.postProcessSegmWin.sigApplyToAllFutureFrames.connect(
                self.postProcessSegmApplyToAllFutureFrames
            )
            self.postProcessSegmWin.show()
            self.postProcessSegmWin.valueChanged(None)
        else:
            self.postProcessSegmWin.close()
            self.postProcessSegmWin = None

    def postProcessSegmApplyToAllFutureFrames(
            self, postProcessKwargs,
            customPostProcessGroupedFeatures,
            customPostProcessFeatures
        ):
        proceed = self.warnEditingWithCca_df(
            'post-processing segmentation', update_images=False
        )
        if not proceed:
            self.logger.info('Post-processing segmentation cancelled.')
            return

        self.progressWin = apps.QDialogWorkerProgress(
            title='Post-processing segmentation', parent=self.host,
            pbarDesc=f'Post-processing segmentation masks...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)

        self.startPostProcessSegmWorker(
            postProcessKwargs, customPostProcessGroupedFeatures,
            customPostProcessFeatures
        )

    def postProcessSegmEditingFinished(self):
        self.update_rp()
        self.store_data()
        self.updateAllImages()

    def postProcessSegmWorkerFinished(self):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None
        self.get_data()
        self.updateAllImages()
        self.titleLabel.setText('Post-processing segmentation done!', color='w')
        self.logger.info('Post-processing segmentation done!')

    def postProcessSegmWinClosed(self):
        self.postProcessSegmWin = None
        self.postProcessSegmAction.toggled.disconnect()
        self.postProcessSegmAction.setChecked(False)
        self.postProcessSegmAction.toggled.connect(self.postProcessSegm)

    def postProcessSegmValueChanged(self, lab, delObjs: dict):
        for delObj in delObjs.values():
            self.clearObjContour(obj=delObj, ax=0)
            self.clearObjContour(obj=delObj, ax=1)

        posData = self.data[self.pos_i]

        labelsToSkip = {}
        for ID in posData.IDs:
            if ID in delObjs:
                labelsToSkip[ID] = True
                continue

            restoreObj = self.postProcessSegmWin.origObjs[ID]
            self.addObjContourToContoursImage(obj=restoreObj, ax=0)
            self.addObjContourToContoursImage(obj=restoreObj, ax=1)

        # self.setAllTextAnnotations(labelsToSkip=labelsToSkip)

        posData.lab = lab
        self.setImageImg2()
        if self.annotSegmMasksCheckbox.isChecked():
            self.labelsLayerImg1.setImage(self.currentLab2D, autoLevels=False)
        if self.annotSegmMasksCheckboxRight.isChecked():
            self.labelsLayerRightImg.setImage(self.currentLab2D, autoLevels=False)

    def resetCursor(self):
        if self.app.overrideCursor() is not None:
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

    def segmFrameCallback(self, action):
        if action == self.addCustomModelFrameAction:
            return

        idx = self.segmActions.index(action)
        model_name = self.modelNames[idx]
        self.repeatSegm(model_name=model_name, askSegmParams=True)

    def showInstructionsCustomModel(self):
        modelFilePath = apps.addCustomModelMessages(self.host)
        if modelFilePath is None:
            self.logger.info('Adding custom model process stopped.')
            return

        self.view_model.store_custom_model_path(modelFilePath)
        modelName = os.path.basename(os.path.dirname(modelFilePath))
        customModelAction = QAction(modelName)
        self.segmSingleFrameMenu.addAction(customModelAction)
        self.segmActions.append(customModelAction)
        self.segmActionsVideo.append(customModelAction)
        self.modelNames.append(modelName)
        self.models.append(None)
        self.sender().callback(customModelAction)

    def reinitStoredSegmModels(self):
        self.models = [None]*len(self.models)

    def segmVideoCallback(self, action):
        if action == self.addCustomModelVideoAction:
            return

        posData = self.data[self.pos_i]
        win = apps.startStopFramesDialog(
            posData.SizeT, currentFrameNum=posData.frame_i+1
        )
        win.exec_()
        if win.cancel:
            self.logger.info('Segmentation on multiple frames aborted.')
            return

        idx = self.segmActionsVideo.index(action)
        model_name = self.modelNames[idx]
        self.repeatSegmVideo(model_name, win.startFrame, win.stopFrame)

    def segmentToolActionTriggered(self):
        if self.segmModelName is None:
            win = apps.QDialogSelectModel(parent=self.host)
            win.exec_()
            if win.cancel:
                self.logger.info('Repeat segmentation cancelled.')
                return
            model_name = win.selectedModel
            self.repeatSegm(
                model_name=model_name, askSegmParams=True
            )
        else:
            self.repeatSegm(model_name=self.segmModelName)

    def initSegmModelParams(
            self, model_name, acdcSegment, init_params, segment_params,
            is_label_roi=False, initLastParams=False,
            extraParams=None, extraParamsTitle=None,ini_filename=None

        ):
        posData = self.data[self.pos_i]
        try:
            url = acdcSegment.url_help()
        except AttributeError:
            url = None

        text_if_cancelled = 'Segmentation process cancelled.'
        out = prompts.init_segm_model_params(
            posData, model_name, init_params, segment_params,
            help_url=url, qparent=self.host, init_last_params=initLastParams,
            check_sam_embeddings=not is_label_roi, is_gui_caller=True,
            extraParams=extraParams,extraParamsTitle=extraParamsTitle,
            ini_filename=ini_filename,
        )
        if out.get('load_sam_embeddings', False):
            self.logger.info('Loading Segment Anything image embeddings...')
            for _posData in self.data:
                _posData.loadSamEmbeddings(logger_func=None)
            text_if_cancelled = 'SAM embeddings loaded.'

        win = out.get('win')
        if win is None:
            self.logger.info(text_if_cancelled)
            self.titleLabel.setText(text_if_cancelled)
            return

        if win.cancel:
            self.logger.info(text_if_cancelled)
            self.titleLabel.setText(text_if_cancelled)
            return

        if model_name != 'thresholding':
            self.model_kwargs = win.model_kwargs

        return win

    @exception_handler
    def repeatSegm(
            self, model_name='', askSegmParams=False, is_label_roi=False
        ):
        model_name = self.view_model.action_model_name(model_name)

        idx = self.modelNames.index(model_name)
        # Ask segm parameters if not already set
        # and not called by segmSingleFrameMenu (askSegmParams=False)
        if not askSegmParams:
            askSegmParams = self.model_kwargs is None

        self.downloadWin = apps.downloadModel(model_name, parent=self.host)
        self.downloadWin.download()

        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        model_name = self.view_model.backend_model_name(model_name)

        posData = self.data[self.pos_i]
        # Check if model needs to be imported
        acdcSegment = self.acdcSegment_li[idx]
        if acdcSegment is None:
            self.logger.info(f'Importing {model_name}...')
            acdcSegment = (
                self.view_model.import_segmentation_module(model_name)
            )
            self.acdcSegment_li[idx] = acdcSegment

        # Ask parameters if the user clicked on the action
        # Otherwise this function is called by "computeSegm" function and
        # we use loaded parameters
        if askSegmParams:
            if self.app.overrideCursor() == Qt.WaitCursor:
                self.app.restoreOverrideCursor()
            self.segmModelName = model_name
            # Read all models parameters
            init_params, segment_params = (
                self.view_model.model_arg_specs(acdcSegment)
            )
            # Prompt user to enter the model parameters
            try:
                url = acdcSegment.url_help()
            except AttributeError:
                url = None

            self.preproc_recipe = None
            initLastParams = True
            if model_name == 'thresholding':
                win = apps.QDialogAutomaticThresholding(
                    parent=self.host, isSegm3D=self.isSegm3D
                )
                win.exec_()
                if win.cancel:
                    return
                self.model_kwargs = win.segment_kwargs
                thresh_method = self.model_kwargs['threshold_method']
                gauss_sigma = self.model_kwargs['gauss_sigma']
                segment_params = (
                    self.view_model.insert_model_arg_spec(
                        segment_params, 'threshold_method', thresh_method
                    )
                )
                segment_params = (
                    self.view_model.insert_model_arg_spec(
                        segment_params, 'gauss_sigma', gauss_sigma
                    )
                )
                initLastParams = False

            win = self.initSegmModelParams(
                model_name, acdcSegment, init_params, segment_params,
                is_label_roi=is_label_roi,
                initLastParams=initLastParams
            )
            if win is None:
                return

            self.standardPostProcessKwargs = win.standardPostProcessKwargs
            self.customPostProcessFeatures = win.customPostProcessFeatures
            self.customPostProcessGroupedFeatures = (
                win.customPostProcessGroupedFeatures
            )
            self.applyPostProcessing = win.applyPostProcessing
            self.secondChannelName = win.secondChannelName
            self.preproc_recipe = win.preproc_recipe

            self.view_model.log_segmentation_params(
                model_name, win.init_kwargs, win.model_kwargs,
                logger_func=self.logger.info,
                preproc_recipe=win.preproc_recipe,
                apply_post_process=self.applyPostProcessing,
                standard_postprocess_kwargs=self.standardPostProcessKwargs,
                custom_postprocess_features=self.customPostProcessFeatures
            )

            use_gpu = win.init_kwargs.get('gpu', False)
            proceed = self.view_model.check_gpu_available(
                model_name, use_gpu, qparent=self.host
            )
            if not proceed:
                self.logger.info('Segmentation process cancelled.')
                self.titleLabel.setText('Segmentation process cancelled.')
                return

            model = self.view_model.init_segmentation_model(
                acdcSegment, posData, win.init_kwargs
            )
            if model is None:
                self.logger.info('Segmentation process cancelled.')
                self.titleLabel.setText('Segmentation process cancelled.')
                return
            try:
                model.setupLogger(self.logger)
            except Exception as e:
                pass
            self.models[idx] = model
            model.model_name = model_name
        else:
            model = self.models[idx]

        if is_label_roi:
            return model

        self.titleLabel.setText(
            f'Segmenting with {model_name}... '
            '(check progress in terminal/console)', color=self.titleColor
        )

        post_process_params = self.view_model.post_process_params(
            apply_postprocessing=self.applyPostProcessing,
            standard_postprocess_kwargs=self.standardPostProcessKwargs,
            custom_postprocess_features=self.customPostProcessFeatures,
        )
        if askSegmParams:
            posData.saveSegmHyperparams(
                model_name, win.init_kwargs, win.model_kwargs,
                post_process_params=post_process_params,
                preproc_recipe=self.preproc_recipe
            )

        if self.askRepeatSegment3D:
            self.segment3D = False
        if self.isSegm3D and self.askRepeatSegment3D:
            msg = widgets.myMessageBox(showCentered=False)
            msg.addDoNotShowAgainCheckbox(text='Do not ask again')
            txt = html_utils.paragraph(
                'Do you want to segment the <b>entire z-stack</b> or only the '
                '<b>current z-slice</b>?'
            )
            _, segment3DButton, _ = msg.question(
                self.host, '3D segmentation?', txt,
                buttonsTexts=(
                    'Cancel', 'Segment 3D z-stack', 'Segment 2D z-slice'
                )
            )
            if msg.cancel:
                self.titleLabel.setText('Segmentation process aborted.')
                self.logger.info('Segmentation process aborted.')
                return
            self.segment3D = msg.clickedButton == segment3DButton
            if msg.doNotShowAgainCheckbox.isChecked():
                self.askRepeatSegment3D = False

        if self.askZrangeSegm3D:
            self.z_range = None
        if self.isSegm3D and self.segment3D and self.askZrangeSegm3D:
            idx = (posData.filename, posData.frame_i)
            try:
                orignal_z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
            except ValueError as e:
                orignal_z = posData.segmInfo_df.loc[idx, 'z_slice_used_gui'].iloc[0]
            selectZtool = apps.QCropZtool(
                posData.SizeZ, parent=self.host, cropButtonText='Ok',
                addDoNotShowAgain=True, title='Select z-slice range to segment'
            )
            selectZtool.sigZvalueChanged.connect(self.selectZtoolZvalueChanged)
            selectZtool.sigCrop.connect(selectZtool.close)
            selectZtool.exec_()
            self.update_z_slice(orignal_z)
            if selectZtool.cancel:
                self.titleLabel.setText('Segmentation process aborted.')
                self.logger.info('Segmentation process aborted.')
                return
            startZ = selectZtool.lowerZscrollbar.value()
            stopZ = selectZtool.upperZscrollbar.value()
            self.z_range = (startZ, stopZ)
            if selectZtool.doNotShowAgainCheckbox.isChecked():
                self.askZrangeSegm3D = False

        secondChannelData = None
        if self.secondChannelName is not None:
            secondChannelData = self.getSecondChannelData()

        self.titleLabel.setText(
            f'{model_name} is thinking... '
            '(check progress in terminal/console)', color=self.titleColor
        )

        self.model = model

        self.segmWorkerMutex = QMutex()
        self.segmWorkerWaitCond = QWaitCondition()
        self.thread = QThread()
        self.worker = workers.segmWorker(
            self.host,
            secondChannelData=secondChannelData,
            mutex=self.segmWorkerMutex,
            waitCond=self.segmWorkerWaitCond
        )
        self.worker.z_range = self.z_range
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        if self.debug:
            self.worker.debug.connect(self.debugSegmWorker)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.segmWorkerFinished)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def debugSegmWorker(self, to_debug):
        img, _lab, lab = to_debug
        printl(img.shape, _lab.shape, lab.shape)
        imshow(img, _lab, lab)
        self.segmWorkerWaitCond.wakeAll()

    def selectZtoolZvalueChanged(self, whichZ, z):
        self.update_z_slice(z)

    @exception_handler
    def repeatSegmVideo(self, model_name, startFrameNum, stopFrameNum):
        model_name = self.view_model.action_model_name(model_name)

        idx = self.modelNames.index(model_name)

        self.downloadWin = apps.downloadModel(model_name, parent=self.host)
        self.downloadWin.download()

        model_name = self.view_model.backend_model_name(model_name)

        posData = self.data[self.pos_i]
        # Check if model needs to be imported
        acdcSegment = self.acdcSegment_li[idx]
        if acdcSegment is None:
            self.logger.info(f'Importing {model_name}...')
            acdcSegment = (
                self.view_model.import_segmentation_module(model_name)
            )
            self.acdcSegment_li[idx] = acdcSegment

        # Read all models parameters
        init_params, segment_params = (
            self.view_model.model_arg_specs(acdcSegment)
        )
        # Prompt user to enter the model parameters
        try:
            url = acdcSegment.url_help()
        except AttributeError:
            url = None

        if model_name == 'thresholding':
            autoThreshWin = apps.QDialogAutomaticThresholding(
                parent=self.host, isSegm3D=self.isSegm3D
            )
            autoThreshWin.exec_()
            if autoThreshWin.cancel:
                return

        win = self.initSegmModelParams(
            model_name, acdcSegment, init_params, segment_params
        )
        if win is None:
            return

        self.standardPostProcessKwargs = win.standardPostProcessKwargs
        self.customPostProcessFeatures = win.customPostProcessFeatures
        self.customPostProcessGroupedFeatures = (
            win.customPostProcessGroupedFeatures
        )
        self.applyPostProcessing = win.applyPostProcessing
        self.preproc_recipe = win.preproc_recipe

        self.view_model.log_segmentation_params(
            model_name, win.init_kwargs, win.model_kwargs,
            logger_func=self.logger.info,
            preproc_recipe=win.preproc_recipe,
            apply_post_process=self.applyPostProcessing,
            standard_postprocess_kwargs=self.standardPostProcessKwargs,
            custom_postprocess_features=self.customPostProcessFeatures
        )

        secondChannelData = None
        if win.secondChannelName is not None:
            secondChannelData = self.getSecondChannelData()

        use_gpu = win.init_kwargs.get('gpu', False)
        proceed = self.view_model.check_gpu_available(
            model_name, use_gpu, qparent=self.host
        )
        if not proceed:
            self.logger.info('Segmentation process cancelled.')
            self.titleLabel.setText('Segmentation process cancelled.')
            return

        model = self.view_model.init_segmentation_model(
            acdcSegment, posData, win.init_kwargs
        )
        if model is None:
            self.logger.info('Segmentation process cancelled.')
            self.titleLabel.setText('Segmentation process cancelled.')
            return
        try:
            model.setupLogger(self.logger)
        except Exception as e:
            pass

        self.extendSegmDataIfNeeded(stopFrameNum)
        self.reInitLastSegmFrame(
            from_frame_i=startFrameNum-1, updateImages=False
        )

        self.titleLabel.setText(
            f'{model_name} is thinking... '
            '(check progress in terminal/console)', color=self.titleColor
        )

        self.progressWin = apps.QDialogWorkerProgress(
            title='Segmenting video', parent=self.host,
            pbarDesc=f'Segmenting from frame n. {startFrameNum} to {stopFrameNum}...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(stopFrameNum-startFrameNum)

        self.thread = QThread()
        self.worker = workers.segmVideoWorker(
            posData, win, model, startFrameNum, stopFrameNum
        )
        self.worker.secondChannelData = secondChannelData
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.segmVideoWorkerFinished)
        self.worker.progressBar.connect(self.workerUpdateProgressbar)
        self.worker.progress.connect(self.workerProgress)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def segmVideoWorkerFinished(self, exec_time):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None

        self.activateAnnotations()

        self.get_data()
        self.tracking(enforce=True)
        self.updateAllImages()

        txt = f'Done. Segmentation computed in {exec_time:.3f} s'
        self.logger.info('-----------------')
        self.logger.info(txt)
        self.logger.info('=================')
        self.titleLabel.setText(txt, color='g')

    def segmWorkerFinished(self, lab, exec_time):
        posData = self.data[self.pos_i]

        if posData.segmInfo_df is not None and posData.SizeZ>1:
            idx = (posData.filename, posData.frame_i)
            posData.segmInfo_df.at[idx, 'resegmented_in_gui'] = True

        if lab.ndim == 2 and self.isSegm3D:
            self.set_2Dlab(lab)
        else:
            posData.lab = lab.copy()

        self.activateAnnotations()

        self.update_rp(wl_update=False)
        self.tracking(enforce=True, against_next=posData.frame_i==0)

        if self.isSnapshot:
            self.fixCcaDfAfterEdit('Repeat segmentation')
            self.updateAllImages()
        else:
            self.warnEditingWithCca_df('Repeat segmentation')

        txt = f'Done. Segmentation computed in {exec_time:.3f} s'
        self.logger.info('-----------------')
        self.logger.info(txt)
        self.logger.info('=================')
        self.titleLabel.setText(txt, color='g')
        self.checkIfAutoSegm()

        QTimer.singleShot(200, self.resizeGui)

    @exception_handler
    def postProcessing(self):
        if self.postProcessSegmWin is None:
            return

        self.postProcessSegmWin.setPosData()
        posData = self.data[self.pos_i]
        lab, delIDs = self.postProcessSegmWin.apply()
        if posData.allData_li[posData.frame_i]['labels'] is None:
            posData.lab = lab.copy()
            self.update_rp()
        else:
            posData.allData_li[posData.frame_i]['labels'] = lab
            self.get_data()

    def checkIfAutoSegm(self):
        """
        If there are any frame or position with empty segmentation mask
        ask whether automatic segmentation should be turned ON
        """
        if self.autoSegmAction.isChecked():
            return
        if self.autoSegmDoNotAskAgain:
            return

        prompt = self.view_model.empty_segmentation_prompt(self.data)
        if not prompt.should_ask:
            return
        txt = prompt.scope_text

        questionTxt = html_utils.paragraph(
            f'Some or all loaded {txt} contain <b>empty segmentation masks</b>.<br><br>'
            'Do you want to <b>activate automatic segmentation</b><sup>*</sup> '
            f'when visiting these {txt}?<br><br>'
            '<i>* Automatic segmentation can always be turned ON/OFF from the menu<br>'
            '  <code>Edit --> Segmentation --> Enable automatic segmentation</code><br><br></i>'
            f'NOTE: you can automatically segment all {txt} using the<br>'
            '    segmentation module.'
        )
        msg = widgets.myMessageBox(wrapText=False)
        noButton, yesButton = msg.question(
            self.host, 'Automatic segmentation?', questionTxt,
            buttonsTexts=('No', 'Yes')
        )
        if msg.clickedButton == yesButton:
            self.autoSegmAction.setChecked(True)
        else:
            self.autoSegmDoNotAskAgain = True
            self.autoSegmAction.setChecked(False)

    def init_segmInfo_df(self):
        for posData in self.data:
            if posData is None:
                # posData is None when computing measurements with the utility
                # and with timelapse data
                continue
            posData.init_segmInfo_df()
