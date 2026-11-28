"""Magic prompts (promptable models) workflow of the main GUI. Mixins of
guiWin (gui/main_window/window.py) only: the measurements viewer never
composes them.
"""
from functools import partial
import pandas as pd
from qtpy.QtCore import QEventLoop, QTimer, QThread
from cellacdc import exception_handler, disableWindow
from cellacdc import prompts, apps, workers, html_utils
from cellacdc import myutils
from cellacdc import widgets
from cellacdc import _warnings
from cellacdc import qutils
from cellacdc.promptable_models import utils as prompt_utils
from cellacdc.promptable_models import saved_models


class MagicPromptsGuiElements:
    """Magic prompts (promptable models): model selection, initialization
    and parameters, point prompts, worker runs, segmentation of the
    image, z-stack or stack, saving and loading of trained models.
    """

    def magicPrompts_cb(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.magicPromptsToolButton)
            self.connectLeftClickButtons()
            self.magicPromptsToolbar.setVisible(True)
            isModelInit = (
                self.magicPromptsToolbar.viewModelParamsAction.isEnabled()
            )
            if not isModelInit:
                # The prompt type depends on the model --> select the
                # model first (prompt type applied once initialised)
                QTimer.singleShot(200, self._magicPromptsSelectModelIfNeeded)
            else:
                self.applyMagicPromptsPromptType()
        else:
            self.brushStrokesEndDrag()
            self.boxPromptsCancelDrag()
            self.drawMagicPromptsBoxes()
            self.resetCursors()
            # Stop selecting cells (hides the selection controls)
            self.magicPromptsToolbar.selectCellsAction.setChecked(False)
            self.applyBrushStrokesSelectCellsMode()
            self.promptSegmentPointsLayerToolbar.setVisible(False)
            self.magicPromptsToolbar.setVisible(False)
            self.brushStrokesLayerImg1.setVisible(False)
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )

    def _magicPromptsSelectModelIfNeeded(self):
        if not self.magicPromptsToolButton.isChecked():
            return
        if self.magicPromptsToolbar.viewModelParamsAction.isEnabled():
            # Model already initialised in the meantime
            return
        self.magicPromptsToolbar.selectModelAction.trigger()

    @disableWindow
    def _importInitMagicPromptModel(
            self, model_name, posData, win, acdcPromptSegment, toolbar
        ):
        self.logger.info(f'Initializing promptable model {model_name}...')
        apps.downloadModel(model_name, parent=self).download()
        init_kwargs = win.init_kwargs
        execution = self.getPromptModelExecution(model_name)
        # `prepare` returns the cached spec if already prepared (it is needed
        # e.g., if the execution was created after loading new data)
        if execution.prepare() is None:
            self.logger.info(
                f'Initialization of {model_name} promptable model cancelled.'
            )
            return
        previous_model = getattr(toolbar, 'model', None)
        model = execution.init_model(
            init_kwargs, posData, replace=previous_model
        )
        if model is None:
            return
        previous_execution = self.promptModelExecutions.get(
             getattr(toolbar, 'model_name', None)
         )
        if previous_execution is not None and previous_execution is not execution:
             previous_execution.release(previous_model)
        toolbar.model = model
        toolbar.model_segment_kwargs = win.model_kwargs
        toolbar.model_name = model_name
        toolbar.viewModelParamsAction.setDisabled(False)
        toolbar.isModelTrained = False
        # Name of the saved model loaded with "Load trained model" (None if
        # the model was trained in this session or not trained)
        toolbar.loadedModelName = None
        # False if the model was trained in this session and not saved
        toolbar.isModelSaved = False
        # Name of the saved model that holds the current model (saved or
        # loaded), to check that it was not removed from the list
        toolbar.savedModelName = None
        self.brushStrokesChangedSinceTraining = True

        self.magicPromptsToolbar.setInitializedModel(
            init_kwargs, toolbar.model_segment_kwargs
        )
        self.magicPromptsToolbar.setSaveLoadModelSupported(
            saved_models.get_saved_model_ext(model) is not None
        )

        prompt_types = tuple(getattr(model, 'prompt_types', ('point',)))
        self.magicPromptsToolbar.blockSignals(True)
        self.magicPromptsToolbar.setSupportedPromptTypes(prompt_types)
        self.magicPromptsToolbar.blockSignals(False)

        self.logger.info(
            f'Promptable model {model_name} successfully initialised '
            f'(supported prompt types: {prompt_types})!'
        )

        self.applyMagicPromptsPromptType()

        # update title
        self.setTitleText()
        return True

    @exception_handler
    @disableWindow
    def magicPromptsInitModel(self, model_name):
        posData = self.data[self.pos_i]
        toolbar = self.magicPromptsToolbar
        execution = self.getPromptModelExecution(model_name)
        model_spec = execution.prepare()
        if model_spec is None:
            self.logger.info(
                f'Initialization of {model_name} promptable model cancelled.'
            )
            return
        init_argspecs, segment_argspecs, help_url = model_spec
        acdcPromptSegment = None

        out = prompts.init_prompt_model_params(
            posData, model_name, init_argspecs, segment_argspecs,
            help_url=help_url, qparent=self, init_last_params=True
        )
        win = out.get('win')
        cancelled = (
            win.cancel
            or not self.magicPromptsConfirmReplaceModel(
                toolbar, f'Initializing {model_name}'
            )
        )
        if cancelled:
            execution.cancel_preparation()
            self.logger.info(
                f'Initialization of {model_name} promptable model cancelled.'
            )
            return

        isInitialized = self._importInitMagicPromptModel(
            model_name, posData, win, acdcPromptSegment, toolbar
        )
        if not isInitialized:
            return

        # Set only now, so that a cancelled selection of another model does
        # not mix its parameters with the initialized model (e.g., in "View
        # model parameters")
        toolbar._model_name = model_name
        toolbar._acdcPromptSegment = acdcPromptSegment
        toolbar._init_argspecs = init_argspecs
        toolbar._segment_argspecs = segment_argspecs
        toolbar._help_url = help_url

    @exception_handler
    def viewSetMagicPromptModelParams(
            self, 
            model_name, 
            acdcPromptSegment,
            init_argspecs, 
            segment_argspecs,
            help_url,
            init_kwargs,
            segment_kwargs,
            toolbar
        ):
        posData = self.data[self.pos_i]
        
        init_argspecs = myutils.setDefaultValueArgSpecsFromKwargs(
            init_argspecs, init_kwargs
        )
        segment_argspecs = myutils.setDefaultValueArgSpecsFromKwargs(
            segment_argspecs, segment_kwargs
        )
        
        out = prompts.init_prompt_model_params(
            posData, model_name, init_argspecs, segment_argspecs, 
            help_url=help_url, qparent=self, init_last_params=False
        )
        win = out.get('win')
        if win.cancel:
            return

        if win.init_kwargs == init_kwargs:
            if win.model_kwargs != segment_kwargs:
                # Only segment parameters changed --> do not re-initialise
                # the model (it would discard e.g., the trained classifier)
                toolbar.model_segment_kwargs = win.model_kwargs
                self.magicPromptsToolbar.setInitializedModel(
                    init_kwargs, toolbar.model_segment_kwargs
                )
                self.logger.info(
                    f'Segmentation parameters of promptable model '
                    f'{model_name} updated.'
                )
            return

        isConfirmed = self.magicPromptsConfirmReplaceModel(
            toolbar, f'Re-initializing {model_name} with the new parameters'
        )
        if not isConfirmed:
            self.logger.info(
                f'Re-initialization of {model_name} promptable model '
                'cancelled.'
            )
            return

        self._importInitMagicPromptModel(
            model_name, posData, win, acdcPromptSegment, toolbar
        )

    def getMagicPromptsInputs(self, toolbar):
        isModelInit = self.magicPromptsToolbar.viewModelParamsAction.isEnabled()
        if not isModelInit:
            _warnings.warnPromptSegmentModelNotInit(qparent=self)
            return

        isPointsLayerInit = (
            self.promptSegmentPointsLayerToolbar.isPointsLayerInit
        )
        df_boxes = self.getMagicPromptsBoxesDf(toolbar)
        if not isPointsLayerInit and df_boxes is None:
            if toolbar.promptType() == 'box':
                _warnings.warnNoBoxPrompts(qparent=self)
            else:
                _warnings.warnPromptSegmentPointsLayerNotInit(qparent=self)
            return

        posData = self.data[self.pos_i]
        image = self.getDisplayedZstack()
        if isPointsLayerInit:
            df_points = self.promptSegmentPointsLayerToolbar.pointsLayerDf(
                posData, isSegm3D=self.isSegm3D
            )
            # Only the points of the segmented frame
            df_points = df_points[df_points['frame_i'] == posData.frame_i]
        else:
            df_points = pd.DataFrame(
                columns=['frame_i', 'Cell_ID', 'z', 'y', 'x', 'id']
            )

        self.logger.info(
            f'Starting {toolbar.model_name} promptable segmentation with the '
            f'following prompts:\n\n{df_points}'
        )
        if df_boxes is not None:
            self.logger.info(f'Box prompts:\n\n{df_boxes}')

        return image, df_points, df_boxes

    @disableWindow
    def magicPromptsComputeOnZoomTriggered(self, toolbar):
        inputs = self.getMagicPromptsInputs(toolbar)
        if inputs is None:
            self.logger.info(
                '"Computing promptable segmentation on zoom" process cancelled.'
            )
            return
        
        posData = self.data[self.pos_i]
        image, df_points, df_boxes = inputs

        ((xmin, xmax), (ymin, ymax)) = self.ax1.viewRange()
        Y, X = image.shape[-2:]
        
        xmin = int(max(0, xmin))
        xmax = int(min(X, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(Y, ymax))
        
        self.logger.info(
            f'Zoom range: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}'
        )
        
        zoom_slice = (slice(ymin, ymax), slice(xmin, xmax))
        
        image = image[..., ymin:ymax, xmin:xmax]
        image_origin = (0, ymin, xmin)
        
        df_points = df_points[df_points['y'] >= ymin]
        df_points = df_points[df_points['x'] >= xmin]
        df_points = df_points[df_points['y'] < ymax]
        df_points = df_points[df_points['x'] < xmax]
        
        df_points['y'] -= ymin
        df_points['x'] -= xmin
        
        df_points = df_points[ df_points['frame_i'] == posData.frame_i]

        if df_boxes is not None:
            df_boxes = self.clipMagicPromptsBoxesToZoom(
                df_boxes, ymin, xmin, ymax, xmax
            )

        self.logger.info(
            f'Image origin = {image_origin}\n'
            f'Image shape = {image.shape}'
        )

        self.startMagicPromptsWorkerAndWait(
            image, df_points, toolbar.model, toolbar.model_segment_kwargs,
            image_origin=image_origin, zoom_slice=zoom_slice,
            df_boxes=df_boxes
        )

    def magicPromptsInterpolateZsliceToggled(self, checked):
        # See 'self.promptSegmentPointsLayerToolbar.addPointsZslicesInterpolation'
        self.promptSegmentPointsLayerToolbar.doAddPointsZslicesInterpolation = (
            checked
        )

    def magicPromptsClearPoints(self, toolbar, only_zoom=False):
        posData = self.data[self.pos_i]
        scatterItem = self.promptSegmentPointsLayerToolbar.scatterItem()
        action = scatterItem.action
        
        pointsDataPos = action.pointsData.get(self.pos_i)
        if pointsDataPos is None:
            return
        
        framePointsData = action.pointsData[self.pos_i].pop(
            posData.frame_i, None
        )
        if framePointsData is None:
            return
        
        if not only_zoom:
            scatterItem.clear()
            return
        
        ((xmin, xmax), (ymin, ymax)) = self.ax1.viewRange()
        Y, X = posData.img_data.shape[-2:]
        
        xmin = int(max(0, xmin))
        xmax = int(min(X, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(Y, ymax))
        
        if 'x' in framePointsData:
            newFramePointsData = {'x': [], 'y': [], 'id': []}
            xx = framePointsData['x']
            yy = framePointsData['y']
            ids = framePointsData['id']
            for x, y, point_id in zip(xx, yy, ids):
                if x < xmin or x >= xmax or y < ymin or y >= ymax:
                    newFramePointsData['x'].append(x)
                    newFramePointsData['y'].append(y)
                    newFramePointsData['id'].append(point_id)
        else:
            newFramePointsData = {}
            for z, zSliceFramePointsData in framePointsData.items():
                newFramePointsData[z] = {'x': [], 'y': [], 'id': []}
                xx = zSliceFramePointsData['x']
                yy = zSliceFramePointsData['y']
                ids = zSliceFramePointsData['id']
                for x, y, point_id in zip(xx, yy, ids):
                    if x < xmin or x >= xmax or y < ymin or y >= ymax:
                        newFramePointsData[z]['x'].append(x)
                        newFramePointsData[z]['y'].append(y)
                        newFramePointsData[z]['id'].append(point_id)
        
        action.pointsData[self.pos_i][posData.frame_i] = newFramePointsData
        self.drawPointsLayers()

    @disableWindow
    def magicPromptsComputeOnImageTriggered(self, toolbar):
        inputs = self.getMagicPromptsInputs(toolbar)
        if inputs is None:
            self.logger.info(
                '"Computing promptable segmentation on entire image" '
                'process cancelled.'
            )
            return

        image, df_points, df_boxes = inputs

        self.startMagicPromptsWorkerAndWait(
            image, df_points, toolbar.model, toolbar.model_segment_kwargs,
            df_boxes=df_boxes
        )

    def startMagicPromptsWorkerAndWait(
            self, image, df_points, model, model_segment_kwargs,
            image_origin=(0, 0, 0), zoom_slice=None, global_image=None,
            id_offset=0, z_slice=None, df_boxes=None
        ):
        desc = (
            'Running promptable segmentation model...'
        )
        self.logger.info(desc)
        posData = self.data[self.pos_i]

        if global_image is None:
            global_image = posData.img_data[posData.frame_i]

        self.progressWin = apps.QDialogWorkerProgress(
            title=desc, parent=self, pbarDesc=desc
        )
        self.progressWin.mainPbar.setMaximum(0)
        self.progressWin.show(self.app)

        self.magicPromptsThread = QThread()
        self.magicPromptsWorker = workers.MagicPromptsWorker(
            posData, image, df_points, model, model_segment_kwargs,
            image_origin=image_origin,
            global_image=global_image,
            id_offset=id_offset,
            z_slice=z_slice,
            df_boxes=df_boxes
        )
        
        self.magicPromptsWorker.moveToThread(
            self.magicPromptsThread
        )
        
        self.magicPromptsWorker.signals.finished.connect(
            self.magicPromptsThread.quit
        )
        self.magicPromptsWorker.signals.finished.connect(
            self.magicPromptsWorker.deleteLater
        )
        self.magicPromptsThread.finished.connect(
            self.magicPromptsThread.deleteLater
        )
        
        self.magicPromptsWorker.signals.critical.connect(
            self.magicPromptsWorkerCritical
        )
        self.magicPromptsWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.magicPromptsWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.magicPromptsWorker.signals.progress.connect(
            self.workerProgress
        )
        self.magicPromptsWorker.signals.finished.connect(
            partial(
                self.magicPromptsWorkerFinished, zoom_slice=zoom_slice,
                image=global_image, z_slice=z_slice
            )
        )
        
        self.magicPromptsThread.started.connect(
            self.magicPromptsWorker.run
        )
        self.magicPromptsThread.start()
        
        self.magicPromptsWorkerLoop = QEventLoop()
        self.magicPromptsWorkerLoop.exec_()

    def magicPromptsWorkerCritical(self, error):
        self.magicPromptsWorkerLoop.exit()
        self.workerCritical(error)

    @exception_handler
    def magicPromptsWorkerFinished(
            self, output, zoom_slice=None, image=None, z_slice=None
        ):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.magicPromptsWorkerLoop.exit()

        lab_new, lab_union, lab_interesection = output

        posData = self.data[self.pos_i]

        is_zoom = True
        if zoom_slice is None:
            zoom_slice = (slice(None), slice(None))
            is_zoom = False

        if image is None:
            image = posData.img_data[posData.frame_i]
        # Labels the output is combined with: the z-slice `z_slice` of the 3D
        # labels if only that z-slice was segmented (a view, edited in place)
        lab = posData.lab
        if z_slice is not None:
            lab = lab[z_slice]
            if image.ndim == lab.ndim + 1:
                image = image[z_slice]
        elif image.ndim == lab.ndim + 1 and image.shape[1:] == lab.shape:
            # 2D masks of z-stack data (e.g., micro-sam projects the masks of
            # the prompted z-slices) --> show the displayed z-slice or a
            # projection (display only)
            if self.zProjComboBox.currentText() == 'single z-slice':
                z = self.zSliceScrollBar.sliderPosition()
                image = image[min(max(int(z), 0), len(image)-1)]
            else:
                image = image.max(axis=0)
        # `image` is the image the model segmented (e.g., the 2D z-slice for
        # 2D segmentation of z-stack data), which matches the labels shape
        img = image[..., zoom_slice[0], zoom_slice[1]]
        images = [img, img, img, img]
        labels_overlays = [
            lab[..., zoom_slice[0], zoom_slice[1]],
            lab_new[..., zoom_slice[0], zoom_slice[1]], 
            lab_union[..., zoom_slice[0], zoom_slice[1]], 
            lab_interesection[..., zoom_slice[0], zoom_slice[1]],
        ]
        labels_overlays_lut = self.getLabelsImageLut()
        labels_overlays_luts = [
            labels_overlays_lut, 
            labels_overlays_lut, 
            labels_overlays_lut,
            labels_overlays_lut,
        ]
        axis_titles = [
            'Original masks', 
            'New masks', 
            'Union of original and new masks',
            'Intersection of original and new masks'
        ]
        
        from cellacdc.plot import imshow
        promptSegmResultsWindow = imshow(
            *images, 
            labels_overlays=labels_overlays,
            labels_overlays_luts=labels_overlays_luts,
            axis_titles=axis_titles,
            window_title='Promptable segmentation results',
            figure_title='Ctrl+Click to select the result to use',
            annotate_labels_idxs=[0, 1, 2, 3],
            selectable_images=True, 
            max_ncols=2,
            lut='gray',
            infer_rgb=False
        )
        if promptSegmResultsWindow.selected_idx is None:
            self.logger.info(
                'Selection of the promptable model segmentation '
                'result cancelled.'
            )
            return
        
        if promptSegmResultsWindow.selected_idx == 0:
            self.logger.info(
                'No selection of a promptable model segmentation '
                'result was made'
            )
            return

        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)
        
        results = (None, lab_new, lab_union, lab_interesection)
        selected_idx = promptSegmResultsWindow.selected_idx
        zoom_out_lab = results[selected_idx][..., zoom_slice[0], zoom_slice[1]]
        zoom_out_lab_mask = zoom_out_lab > 0

        # Write directly into current labels (writing into the stored
        # labels and calling `get_data` would discard unsaved edits and
        # fail if the frame was never stored). `lab` is `posData.lab` or a
        # view of one of its z-slices
        lab[..., zoom_slice[0], zoom_slice[1]][zoom_out_lab_mask] = (
            zoom_out_lab[zoom_out_lab_mask]
        )

        self.update_rp()
        self.store_data(autosave=False)
        self.updateAllImages()

    def magicPromptsPromptTypeChanged(self, toolbar, prompt_type):
        self.applyMagicPromptsPromptType()

    def applyMagicPromptsPromptType(self):
        brushCircles = (self.ax2_BrushCircle, self.ax1_BrushCircle)
        # Hide the "Select cells" controls if not selecting cells anymore
        # (e.g., model initialization switches to points while the toolbar
        # signals are blocked)
        self.applyBrushStrokesSelectCellsMode()
        if not self.magicPromptsToolButton.isChecked():
            self.brushStrokesLayerImg1.setVisible(False)
            self.boxPromptsCancelDrag()
            self.drawMagicPromptsBoxes()
            return

        prompt_type = self.magicPromptsToolbar.promptType()
        if prompt_type != 'box':
            self.boxPromptsCancelDrag()
        # Boxes are displayed with points and boxes prompt types (editable
        # only with boxes)
        self.drawMagicPromptsBoxes()
        if prompt_type == 'brush_stroke':
            self.promptSegmentPointsLayerToolbar.setVisible(False)
            # Deactivate adding prompt points (e.g., middle-click would add
            # points instead of deleting objects, and Ctrl+Z would undo
            # hidden points)
            self.deactivatePromptPointsLayerClickButtons()
            # Neutral colours for the brush circle (the points layer
            # sets the circle colours to the layer colour)
            for brushCircle in brushCircles:
                brushCircle.setPen(self.ax2_BrushCirclePen)
                brushCircle.setBrush(self.ax2_BrushCircleBrush)
            # "Segment z-stack" only for z-stack data
            try:
                isZstack = self.data[self.pos_i].SizeZ > 1
            except (AttributeError, IndexError):
                isZstack = True
            self.magicPromptsToolbar.setZstackData(isZstack)
            self.setBrushStrokesLayerImage()
            return

        self.brushStrokesEndDrag()
        self.brushStrokesLayerImg1.setVisible(False)
        self.setHoverToolSymbolData([], [], brushCircles)
        self.reactivatePromptPointsLayerClickButtons()
        if prompt_type == 'box':
            # The points stay visible (they are combined with the boxes) and
            # their "Left-click ID" and "Right-click ID" are used for the
            # boxes too, but clicks draw boxes
            self.promptSegmentPointsLayerToolbar.setVisible(
                self.promptSegmentPointsLayerToolbar.isPointsLayerInit
            )
            self.gui_refreshToolCursor()
            return

        addPointsByClickingButton = self.buttonAddPointsByClickingActive()
        if addPointsByClickingButton is not None:
            action = addPointsByClickingButton.action
            self.ax1_BrushCircle.setBrush(action.brushColor)
            self.ax1_BrushCircle.setPen(action.penColor)
            # Keep the points undo available when coming back from the
            # boxes or brush strokes prompt types (their undo may have
            # disabled the undo action)
            undoQueues = getattr(self, 'undoAddPointQueueMapper', None) or {}
            if undoQueues.get(action):
                self.undoAction.setEnabled(True)
        self.promptSegmentPointsLayerToolbar.setVisible(True)
        if not self.promptSegmentPointsLayerToolbar.isPointsLayerInit:
            # Deferred so that the (non-modal) dialog is not opened while
            # the window is disabled (e.g., during model initialization)
            QTimer.singleShot(100, self._magicPromptsAddPointsLayerIfNeeded)

    def promptPointsLayerClickButtons(self):
        """Buttons to add points by clicking of the promptable models
        points layer"""
        buttons = []
        for action in self.promptSegmentPointsLayerToolbar.actions()[1:]:
            if getattr(action, 'layerTypeIdx', None) != 4:
                continue
            buttons.append(action.button)
        return buttons

    def deactivatePromptPointsLayerClickButtons(self):
        toRecheck = getattr(self, 'promptPointsButtonsToRecheck', [])
        for button in self.promptPointsLayerClickButtons():
            if not button.isChecked():
                continue
            button.setChecked(False)
            # Hide points of the hidden layer (also done by
            # `addPointsByClickingButtonToggled` when connected)
            button.action.scatterItem.setVisible(False)
            if button not in toRecheck:
                toRecheck.append(button)
        self.promptPointsButtonsToRecheck = toRecheck

    def reactivatePromptPointsLayerClickButtons(self):
        toRecheck = getattr(self, 'promptPointsButtonsToRecheck', [])
        self.promptPointsButtonsToRecheck = []
        validButtons = self.promptPointsLayerClickButtons()
        for button in toRecheck:
            if button not in validButtons or button.isChecked():
                continue
            # Same as clicking on the button (`addPointsByClickingButtonToggled`)
            button.setChecked(True)
            button.action.scatterItem.setVisible(True)

    def _magicPromptsAddPointsLayerIfNeeded(self):
        if not self.magicPromptsToolButton.isChecked():
            return
        if self.magicPromptsToolbar.promptType() != 'point':
            return
        if self.promptSegmentPointsLayerToolbar.isPointsLayerInit:
            return
        self.addPointsLayerTriggered(
            toolbar=self.promptSegmentPointsLayerToolbar
        )

    def magicPromptsPointsLayerButton(self):
        """Button of the prompts points layer with "Left-click ID" and
        "Right-click ID" controls (None if the layer is not initialized)"""
        buttons = self.promptPointsLayerClickButtons()
        if not buttons:
            return
        return buttons[0]

    def magicPromptsUsedPromptIds(self):
        """IDs of the point and box prompts of the current frame"""
        posData = self.data[self.pos_i]
        ids = {box['id'] for box in self.getMagicPromptsBoxes()}
        button = self.magicPromptsPointsLayerButton()
        if button is None:
            return ids
        pointsDataPos = button.action.pointsData.get(self.pos_i, {})
        framePointsData = pointsDataPos.get(posData.frame_i)
        if not framePointsData:
            return ids
        if 'x' in framePointsData:
            ids.update(framePointsData['id'])
        else:
            for zSlicePointsData in framePointsData.values():
                ids.update(zSlicePointsData['id'])
        ids.discard(0)
        return ids

    def magicPromptsNewObjectId(self):
        """New object ID not used by the masks nor by the prompts"""
        new_id = self.setBrushID(return_val=True)
        used_ids = self.magicPromptsUsedPromptIds()
        if used_ids:
            new_id = max(new_id, max(used_ids)+1)
        return int(new_id)

    def isMasksUndoAvailable(self):
        """True if `undo` can restore a previous state of the masks of the
        current frame"""
        posData = self.data[self.pos_i]
        try:
            states = posData.UndoRedoStates[posData.frame_i]
        except Exception as err:
            return False
        if self.UndoCount == 0:
            # `undo` stores the current state first
            return len(states) > 0
        return self.UndoCount < len(states) - 1

    def askRetrainLoadedPromptModel(self, model_name, loadedModelName):
        """Ask whether to replace the loaded model by re-training it on the
        brush strokes. Returns 'retrain', 'use_loaded', or 'cancel'."""
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            The current {model_name} model was <b>loaded</b> from the saved
            model <code>{loadedModelName}</code> and the brush strokes
            changed since then.<br><br>
            <b>Re-training</b> creates a new model trained <b>only on the
            current brush strokes</b><br>
            (the loaded model is replaced, but its saved file is not
            modified).<br><br>
            What do you want to do?
        """)
        _, useLoadedButton, retrainButton = msg.question(
            self, 'Re-train loaded model?', txt,
            buttonsTexts=(
                'Cancel', 'Use loaded model', 'Re-train on brush strokes'
            )
        )
        if msg.clickedButton == retrainButton:
            return 'retrain'
        if msg.clickedButton == useLoadedButton:
            return 'use_loaded'
        return 'cancel'

    def magicPromptsRunModelTask(self, desc, func, *args, **kwargs):
        """Run `func(*args, **kwargs)` (e.g., training or loading a model)
        in a background thread while showing a busy progress window, and
        return its result."""
        from cellacdc.service import model_execution

        self.titleLabel.setText(desc, color=self.titleColor)
        progressWin = apps.QDialogWorkerProgress(
            title=desc, parent=self, pbarDesc=desc
        )
        progressWin.mainPbar.setMaximum(0)
        progressWin.show(self.app)
        wasEnabled = self.isEnabled()
        self.setDisabled(True)
        try:
            return model_execution.run_in_background(
                self, func, *args, **kwargs
            )
        finally:
            progressWin.workerFinished = True
            progressWin.close()
            if wasEnabled:
                self.setDisabled(False)

    def magicPromptsSavedModelExt(self, toolbar):
        """Extension of the saved model files of the toolbar's model, or
        None (with a warning) if the model is not initialized or cannot be
        saved."""
        if not toolbar.viewModelParamsAction.isEnabled():
            _warnings.warnPromptSegmentModelNotInit(qparent=self)
            return None

        ext = saved_models.get_saved_model_ext(toolbar.model)
        if ext is None:
            self.logger.info(
                f'{toolbar.model_name} cannot save or load trained models.'
            )
        return ext

    @exception_handler
    @disableWindow
    def magicPromptsSaveModelTriggered(self, toolbar):
        self.magicPromptsSaveModel(toolbar)

    def magicPromptsSaveModel(self, toolbar, retrain=True):
        """Ask for a name and save the trained model in the saved models
        folder. Returns the `saved_models.SavedModel` (None if not saved).

        If `retrain` is True, the model is (re-)trained first if the brush
        strokes changed since the last training (like "Segment image"),
        otherwise the current trained model is saved as is.
        """
        ext = self.magicPromptsSavedModelExt(toolbar)
        if ext is None:
            return None

        model_name = toolbar.model_name
        if retrain:
            isReady = self.magicPromptsEnsureBrushStrokesTrained(toolbar)
        else:
            isReady = getattr(toolbar, 'isModelTrained', False)
        if not isReady:
            self.logger.info(f'Saving {model_name} model cancelled.')
            return None

        folderpath = saved_models.get_folderpath(model_name)
        helpText = (f"""
            The model is saved in the following folder:<br><br>
            <code>{folderpath}</code><br><br>
            You can load it later (e.g., to segment other data) with the
            "<b>Load trained model</b>" button of the promptable models
            toolbar.
        """)
        win = apps.filenameDialog(
            ext=ext,
            title=f'Save trained {model_name} model',
            hintText=html_utils.paragraph(
                f'Insert a <b>name</b> for the trained {model_name} model:'
            ),
            existingNames=saved_models.get_saved_model_names(
                model_name, ext
            ),
            # Model files names are case-insensitive on Windows and macOS
            caseInsensitiveNames=True,
            allowEmpty=False,
            helpText=helpText,
            parent=self
        )
        win.exec_()
        if win.cancel:
            self.logger.info(f'Saving {model_name} model cancelled.')
            return None

        name = saved_models.sanitize_name(win.filename[:-len(ext)])
        # Saved in a staging folder first: an existing model with the same
        # name is replaced only if saving succeeds
        stagedFilepath = saved_models.prepare_save(model_name, name, ext)
        try:
            self.magicPromptsRunModelTask(
                f'Saving {model_name} model "{name}"...',
                toolbar.model.save_model, stagedFilepath
            )
        except Exception:
            saved_models.cancel_save(stagedFilepath)
            raise
        savedModel = saved_models.finish_save(model_name, stagedFilepath)

        toolbar.isModelSaved = True
        toolbar.savedModelName = savedModel.name
        self.logger.info(
            f'Trained {model_name} model "{name}" saved to '
            f'"{savedModel.filepath}"'
        )
        self.titleLabel.setText(
            f'Trained {model_name} model "{name}" saved!', color='g'
        )
        return savedModel

    def magicPromptsRefreshSavedState(self, toolbar, ext):
        """The model is not saved anymore if its saved model was removed
        from the list (e.g., with "Remove from list" in the load dialog)."""
        if not getattr(toolbar, 'isModelSaved', False):
            return

        name = getattr(toolbar, 'savedModelName', None)
        savedModel = None
        if name is not None:
            savedModel = saved_models.get_saved_model(
                toolbar.model_name, name, ext
            )
        if savedModel is None or not savedModel.exists:
            toolbar.isModelSaved = False
            toolbar.savedModelName = None

    def askReplaceUnsavedPromptModel(self, model_name, action):
        """Ask what to do with the trained model (not saved) before `action`
        (e.g., 'Loading another model') replaces it. Returns 'save',
        'replace', or 'cancel'."""
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            The current {model_name} model was <b>trained</b> in this
            session and it was <b>not saved</b>.<br><br>
            {action} <b>replaces</b> it.<br><br>
            Do you want to save the current model first?
        """)
        _, replaceButton, saveButton = msg.warning(
            self, 'Save current model?', txt,
            buttonsTexts=(
                'Cancel', 'Continue without saving',
                'Save current model first...'
            )
        )
        if msg.clickedButton == saveButton:
            return 'save'
        if msg.clickedButton == replaceButton:
            return 'replace'
        return 'cancel'

    def magicPromptsConfirmReplaceModel(self, toolbar, action):
        """Before `action` replaces the current model: if it was trained and
        not saved, ask whether to save it first. Returns False if the action
        must be cancelled, otherwise the saved model (or True if there was
        nothing to save)."""
        ext = saved_models.get_saved_model_ext(getattr(toolbar, 'model', None))
        if ext is None:
            return True

        self.magicPromptsRefreshSavedState(toolbar, ext)
        isUnsaved = (
            getattr(toolbar, 'isModelTrained', False)
            and not getattr(toolbar, 'isModelSaved', False)
        )
        if not isUnsaved:
            return True

        answer = self.askReplaceUnsavedPromptModel(toolbar.model_name, action)
        if answer == 'replace':
            return True
        if answer == 'save':
            # Save the trained model as is (no re-training)
            savedModel = self.magicPromptsSaveModel(toolbar, retrain=False)
            if savedModel is not None:
                return savedModel
        return False

    def warnLoadedPromptModelClasses(self, model_name, name, classes):
        """Warn if the loaded classifier does not have exactly the classes
        1 (background) and 2 (foreground), e.g., trained in napari with more
        classes."""
        if classes is None or set(classes) == {1, 2}:
            return

        classes_str = ', '.join(str(c) for c in classes)
        if 2 in classes:
            details = (
                'Cell-ACDC uses only class <b>2 as foreground</b>, all the '
                'other classes are considered background.'
            )
        else:
            details = (
                'The model does <b>not have the class 2</b>, which Cell-ACDC '
                'uses as foreground, so <b>no objects will be segmented</b>.'
            )
        self.logger.warning(
            f'The loaded {model_name} model "{name}" has the classes '
            f'{classes_str} (Cell-ACDC uses class 2 as foreground).'
        )
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            The loaded {model_name} model <code>{name}</code> has the
            classes <code>{classes_str}</code>.<br><br>
            {details}
        """)
        msg.warning(self, 'Model classes', txt)

    @exception_handler
    @disableWindow
    def magicPromptsLoadModelTriggered(self, toolbar):
        ext = self.magicPromptsSavedModelExt(toolbar)
        if ext is None:
            return

        model_name = toolbar.model_name
        win = apps.LoadSavedPromptableModelDialog(
            model_name, ext, start_dir=myutils.getMostRecentPath(),
            parent=self
        )
        win.exec_()
        # The saved model of the current model may have been removed
        self.magicPromptsRefreshSavedState(toolbar, ext)
        if win.cancel:
            self.logger.info(f'Loading trained {model_name} model cancelled.')
            return

        savedModel = win.selectedModel
        confirmed = self.magicPromptsConfirmReplaceModel(
            toolbar, 'Loading another model'
        )
        if not confirmed:
            self.logger.info(f'Loading trained {model_name} model cancelled.')
            return

        isSelectedOverwritten = (
            isinstance(confirmed, saved_models.SavedModel)
            and not win.isNewLink
            and confirmed.name.casefold() == savedModel.name.casefold()
        )
        if isSelectedOverwritten:
            # The current model was saved with the name of the model
            # selected for loading --> the selected model does not exist
            # anymore and loading it would load the current model again
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(f"""
                The current model was saved as <code>{confirmed.name}</code>,
                replacing the model selected for loading.<br><br>
                Loading cancelled.
            """)
            msg.warning(self, 'Selected model replaced', txt)
            self.logger.info(f'Loading trained {model_name} model cancelled.')
            return

        self.logger.info(
            f'Loading trained {model_name} model "{savedModel.name}" from '
            f'"{savedModel.filepath}"...'
        )
        # The model will not match its init parameters anymore --> its
        # worker (isolated environment) must not be reused for them (also if
        # loading fails half-way)
        execution = self.promptModelExecutions.get(model_name)
        if execution is not None:
            execution.forget_init_kwargs(toolbar.model)

        params = self.magicPromptsRunModelTask(
            f'Loading {model_name} model "{savedModel.name}"...',
            toolbar.model.load_model, savedModel.filepath
        )

        if win.isNewLink:
            # Selected with "Browse..." --> add it to the list as a link
            # now that it loaded successfully
            try:
                savedModel = saved_models.add_link(
                    model_name, savedModel.filepath, ext
                )
            except Exception as err:
                self.logger.warning(
                    f'The model file "{savedModel.filepath}" could not be '
                    f'added to the list of saved models: {err}'
                )

        toolbar.isModelTrained = True
        toolbar.isModelSaved = True
        toolbar.savedModelName = savedModel.name
        toolbar.loadedModelName = savedModel.name
        # The loaded model is used as is until new brush strokes are painted
        self.brushStrokesChangedSinceTraining = False

        # Show the parameters of the loaded model in "View model parameters"
        init_kwargs = dict(getattr(toolbar, '_init_kwargs', None) or {})
        if isinstance(params, dict):
            for key, value in params.items():
                if key in init_kwargs and value is not None:
                    init_kwargs[key] = value
        toolbar.setInitializedModel(init_kwargs, toolbar.model_segment_kwargs)

        self.logger.info(
            f'Trained {model_name} model "{savedModel.name}" loaded. '
            f'Model parameters: {params}'
        )
        isCustomFeatureExtractor = (
            isinstance(params, dict)
            and 'feature_extractor' in init_kwargs
            and params.get('feature_extractor', '') is None
        )
        if isCustomFeatureExtractor:
            self.logger.warning(
                f'The feature extractor of the loaded model '
                f'("{params.get("fe_name")}", layers {params.get("fe_layers")}) '
                'is not one of the options of "View model parameters", '
                'which shows the previous feature extractor. Changing the '
                'initialization parameters there creates a new model with '
                'the feature extractor shown.'
            )
        self.titleLabel.setText(
            f'Trained {model_name} model "{savedModel.name}" loaded!',
            color='g'
        )
        if isinstance(params, dict):
            self.warnLoadedPromptModelClasses(
                model_name, savedModel.name, params.get('classes')
            )

    @exception_handler
    @disableWindow
    def magicPromptsSegmentImageTriggered(self, toolbar):
        """Segment the displayed image with the model trained on brush
        strokes: the current frame with 2D masks, and only the displayed
        z-slice with 3D masks (see `magicPromptsSegmentZstackTriggered` to
        segment all the z-slices)."""
        posData = self.data[self.pos_i]
        z_slice = None
        if posData.lab.ndim == 3:
            if not self.isBrushStrokesViewPaintable(3):
                _warnings.warnSegmentImageRequiresZslice(qparent=self)
                self.logger.info('Segmentation with brush strokes cancelled.')
                return
            z_slice = self.zSliceScrollBar.sliderPosition()
            z_slice = min(max(z_slice, 0), len(posData.lab)-1)

        isReady = self.magicPromptsEnsureBrushStrokesTrained(toolbar)
        if not isReady:
            self.logger.info('Segmentation with brush strokes cancelled.')
            return

        self.magicPromptsSegmentCurrentFrame(toolbar, z_slice=z_slice)

    @exception_handler
    @disableWindow
    def magicPromptsSegmentZstackTriggered(self, toolbar):
        """Segment all the z-slices of the current frame with the model
        trained on brush strokes (3D segmentation masks only)."""
        posData = self.data[self.pos_i]
        isZstack = getattr(posData, 'SizeZ', 1) > 1
        if isZstack and posData.lab.ndim != 3:
            _warnings.warnSegmentZstackRequires3DMasks(qparent=self)
            self.logger.info('Segmentation of the z-stack cancelled.')
            return

        if not isZstack:
            self.logger.info(
                'Loaded data is not a z-stack --> segmenting the current '
                'image.'
            )

        isReady = self.magicPromptsEnsureBrushStrokesTrained(toolbar)
        if not isReady:
            self.logger.info('Segmentation with brush strokes cancelled.')
            return

        self.magicPromptsSegmentCurrentFrame(toolbar)

    def magicPromptsSegmentCurrentFrame(self, toolbar, z_slice=None):
        """Segment the current frame (only the z-slice `z_slice` of a
        z-stack, if not None) with the trained model and show the results
        to choose how to combine them with the existing masks."""
        posData = self.data[self.pos_i]
        image = prompt_utils.get_brush_strokes_image(
            posData, posData.frame_i, posData.lab.ndim
        )
        # New IDs from skimage.measure.label start at 1 --> offset them to
        # avoid collisions with existing IDs
        id_offset = self.setBrushID(return_val=True) - 1

        if z_slice is not None:
            what = f'z-slice n. {z_slice+1} of the current frame'
        elif image.ndim == 3:
            what = 'z-stack of the current frame'
        else:
            what = 'current image'
        self.logger.info(f'Segmenting {what} with {toolbar.model_name}...')
        self.startMagicPromptsWorkerAndWait(
            image, None, toolbar.model, toolbar.model_segment_kwargs,
            global_image=image, id_offset=id_offset, z_slice=z_slice
        )

    @exception_handler
    def magicPromptsSegmentStackTriggered(self, toolbar):
        # The dialogs of this flow do not block the main window --> disable
        # the toolbar (e.g., "Load trained model") until the frames are
        # segmented (re-enabled by `_magicPromptsSegmStackDone`)
        self.magicPromptsToolbar.setDisabled(True)
        isStarted = False
        try:
            isStarted = self._magicPromptsSegmentStack(toolbar)
        finally:
            if not isStarted:
                self.magicPromptsToolbar.setDisabled(False)

    def _magicPromptsSegmentStack(self, toolbar):
        """Returns True if the worker segmenting the frames was started."""
        posData = self.data[self.pos_i]
        if posData.SizeT == 1:
            self.logger.info(
                'Loaded data has only one frame --> segmenting only the '
                'current frame.'
            )
            # Entire z-stack with 3D masks, the 2D image otherwise
            if posData.lab.ndim == 3:
                self.magicPromptsSegmentZstackTriggered(toolbar)
            else:
                self.magicPromptsSegmentImageTriggered(toolbar)
            return

        isReady = self.magicPromptsEnsureBrushStrokesTrained(toolbar)
        if not isReady:
            self.logger.info('Segmentation with brush strokes cancelled.')
            return

        model_name = toolbar.model_name
        win = apps.startStopFramesDialog(
            posData.SizeT, currentFrameNum=posData.frame_i+1, parent=self,
            windowTitle=(
                f'Select frame range to segment with {model_name} '
                '(existing masks will be replaced)'
            )
        )
        win.exec_()
        if win.cancel:
            self.logger.info('Segmentation of multiple frames cancelled.')
            return

        startFrameNum, stopFrameNum = win.startFrame, win.stopFrame

        # Never re-initialize past the current frame: the frames after the
        # last validated frame (incl. the first segmented one when starting
        # after the current frame) are then tracked when visited
        from_frame_i = min(startFrameNum-1, posData.frame_i)
        cancel = self.warnReinitLastSegmFrame(frame_n=from_frame_i+1)
        if cancel:
            self.logger.info('Segmentation of multiple frames cancelled.')
            return

        # Store current edits since `segmVideoWorkerFinished` calls get_data
        self.store_data(autosave=False)
        self.extendSegmDataIfNeeded(stopFrameNum)
        self.reInitLastSegmFrame(
            from_frame_i=from_frame_i, updateImages=False, force=True
        )

        # `segmVideoWorkerFinished` releases `self.segmVideoModel` --> make
        # sure it is not the promptable model (and release stale models)
        previous = getattr(self, 'segmVideoModel', None)
        if previous is not None:
            self.segmVideoModel = None
            self.releaseSegmModel(*previous)

        annot_ndim = len(self.getBrushStrokesAnnotShape(posData))

        self.titleLabel.setText(
            f'{model_name} is segmenting frames... '
            '(check progress in terminal/console)', color=self.titleColor
        )

        self.progressWin = apps.QDialogWorkerProgress(
            title=f'Segmenting frames with {model_name}', parent=self,
            pbarDesc=(
                f'Segmenting from frame n. {startFrameNum} to '
                f'{stopFrameNum}...'
            )
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(stopFrameNum-startFrameNum+1)

        self.magicPromptsSegmStackThread = QThread()
        self.magicPromptsSegmStackWorker = (
            workers.MagicPromptsSegmStackWorker(
                posData, toolbar.model, toolbar.model_segment_kwargs,
                startFrameNum, stopFrameNum, annot_ndim
            )
        )
        worker = self.magicPromptsSegmStackWorker
        thread = self.magicPromptsSegmStackThread
        worker.moveToThread(thread)

        # The worker uses the toolbar's model --> prevent training or
        # segmenting (or re-initializing the model) while it runs. Connected
        # first so that the toolbar is re-enabled even if the next slots fail
        worker.finished.connect(self._magicPromptsSegmStackDone)
        worker.critical.connect(self._magicPromptsSegmStackDone)

        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        worker.critical.connect(self.workerCritical)
        worker.finished.connect(self.segmVideoWorkerFinished)
        worker.progressBar.connect(self.workerUpdateProgressbar)
        worker.progress.connect(self.workerProgress)

        thread.started.connect(worker.run)
        self.magicPromptsToolbar.setDisabled(True)
        thread.start()
        return True

    def _magicPromptsSegmStackDone(self, *args):
        self.magicPromptsToolbar.setDisabled(False)

    def showInstructionsCustomPromptModel(self):
        modelFilePath = apps.addCustomPromptModelMessages(QParent=self)
        if modelFilePath is None:
            self.logger.info('Adding custom promptable model process stopped.')
            return
        
        myutils.store_custom_promptable_model_path(modelFilePath)
        
        msg = widgets.myMessageBox(wrapText=False)
        info_txt = html_utils.paragraph(f"""
            Done!<br><br>
            The custom promptable model has been added to the list of models.<br><br>
            Use the <b>Magic prompts</b> button (top toolbar) to use it.<br><br>
            Have fun!
        """)
        msg.information(self, 'Custom promptable model added', info_txt)

    def segmWithPromptableModelActionTriggered(self):
        blinker = qutils.QControlBlink(
            self.magicPromptsToolButton, qparent=self
        )
        blinker.start()
