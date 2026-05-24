"""Qt view adapter for promptable segmentation workflows."""

from __future__ import annotations

from functools import partial

from qtpy.QtCore import QEventLoop, QThread

from cellacdc import (
    _warnings,
    apps,
    exception_handler,
    html_utils,
    prompts,
    qutils,
    widgets,
    workers,
)
from cellacdc import disableWindow
from cellacdc.viewmodels.magic_prompts_viewmodel import MagicPromptsViewModel


class MagicPromptsView:
    """Qt-facing adapter around promptable segmentation dialogs and workers."""

    def __init__(self, host, view_model: MagicPromptsViewModel):
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'view_model', view_model)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host', 'view_model'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def showInstructionsCustomPromptModel(self):
        modelFilePath = apps.addCustomPromptModelMessages(QParent=self.host)
        if modelFilePath is None:
            self.logger.info('Adding custom promptable model process stopped.')
            return

        self.view_model.store_custom_promptable_model_path(
            modelFilePath
        )

        msg = widgets.myMessageBox(wrapText=False)
        info_txt = html_utils.paragraph(f"""
            Done!<br><br>
            The custom promptable model has been added to the list of models.<br><br>
            Use the <b>Magic prompts</b> button (top toolbar) to use it.<br><br>
            Have fun!
        """)
        msg.information(self.host, 'Custom promptable model added', info_txt)

    def segmWithPromptableModelActionTriggered(self):
        self.blinker = qutils.QControlBlink(
            self.magicPromptsToolButton, qparent=self.host
        )
        self.blinker.start()

    @disableWindow
    def _importInitMagicPromptModel(
            self, model_name, posData, win, acdcPromptSegment, toolbar
        ):
        self.logger.info(f'Initializing promptable model {model_name}...')
        init_kwargs = win.init_kwargs
        model = self.view_model.init_prompt_segmentation_model(
            acdcPromptSegment, posData, win.init_kwargs
        )
        toolbar.model = model
        toolbar.model_segment_kwargs = win.model_kwargs
        toolbar.model_name = model_name
        toolbar.viewModelParamsAction.setDisabled(False)

        self.magicPromptsToolbar.setInitializedModel(
            init_kwargs, toolbar.model_segment_kwargs
        )

        self.logger.info(
            f'Promptable model {model_name} successfully initialised!'
        )

    @exception_handler
    def magicPromptsInitModel(
            self,
            model_name,
            acdcPromptSegment,
            init_argspecs,
            segment_argspecs,
            help_url,
            toolbar,
        ):
        posData = self.data[self.pos_i]

        out = prompts.init_prompt_model_params(
            posData, model_name, init_argspecs, segment_argspecs,
            help_url=help_url, qparent=self.host, init_last_params=True
        )
        win = out.get('win')
        if win.cancel:
            self.logger.info(
                f'Initialization of {model_name} promptable model cancelled.'
            )
            return

        self._importInitMagicPromptModel(
            model_name, posData, win, acdcPromptSegment, toolbar
        )

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

        init_argspecs = (
            self.view_model.set_default_arg_specs_from_kwargs(
                init_argspecs, init_kwargs
            )
        )
        segment_argspecs = (
            self.view_model.set_default_arg_specs_from_kwargs(
                segment_argspecs, segment_kwargs
            )
        )

        out = prompts.init_prompt_model_params(
            posData, model_name, init_argspecs, segment_argspecs,
            help_url=help_url, qparent=self.host, init_last_params=False
        )
        win = out.get('win')
        if win.cancel:
            return

        if win.model_kwargs != segment_kwargs or win.init_kwargs != init_kwargs:
            self._importInitMagicPromptModel(
                model_name, posData, win, acdcPromptSegment, toolbar
            )

    def getMagicPromptsInputs(self, toolbar):
        if not self.promptSegmentPointsLayerToolbar.isPointsLayerInit:
            _warnings.warnPromptSegmentPointsLayerNotInit(qparent=self.host)
            return

        if not self.magicPromptsToolbar.viewModelParamsAction.isEnabled():
            _warnings.warnPromptSegmentModelNotInit(qparent=self.host)
            return

        posData = self.data[self.pos_i]
        image = self.getDisplayedZstack()
        df_points = self.promptSegmentPointsLayerToolbar.pointsLayerDf(
            posData, isSegm3D=self.isSegm3D
        )

        self.logger.info(
            f'Starting {toolbar.model_name} promptable segmentation with the '
            f'following prompts:\n\n{df_points}'
        )

        return image, df_points

    @disableWindow
    def magicPromptsComputeOnZoomTriggered(self, toolbar):
        inputs = self.getMagicPromptsInputs(toolbar)
        if inputs is None:
            self.logger.info(
                '"Computing promptable segmentation on zoom" process cancelled.'
            )
            return

        posData = self.data[self.pos_i]
        image, df_points = inputs

        zoom = self.view_model.zoom_region(self.ax1.viewRange(), image.shape)
        xmin, xmax, ymin, ymax = zoom.bounds

        self.logger.info(
            f'Zoom range: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}'
        )

        zoom_slice = zoom.zoom_slice
        image = image[..., zoom_slice[0], zoom_slice[1]]
        image_origin = zoom.image_origin
        df_points = self.view_model.points_in_zoom(
            df_points,
            zoom,
            posData.frame_i,
        )

        self.logger.info(
            f'Image origin = {image_origin}\n'
            f'Image shape = {image.shape}'
        )

        self.startMagicPromptsWorkerAndWait(
            image, df_points, toolbar.model, toolbar.model_segment_kwargs,
            image_origin=image_origin, zoom_slice=zoom_slice
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

        zoom = self.view_model.zoom_region(
            self.ax1.viewRange(),
            posData.img_data.shape,
        )
        newFramePointsData = self.view_model.retained_points_outside_zoom(
            framePointsData,
            zoom,
        )

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

        image, df_points = inputs

        self.startMagicPromptsWorkerAndWait(
            image, df_points, toolbar.model, toolbar.model_segment_kwargs
        )

    def startMagicPromptsWorkerAndWait(
            self, image, df_points, model, model_segment_kwargs,
            image_origin=(0, 0, 0), zoom_slice=None
        ):
        desc = (
            'Running promptable segmentation model...'
        )
        self.logger.info(desc)
        posData = self.data[self.pos_i]

        self.progressWin = apps.QDialogWorkerProgress(
            title=desc, parent=self.host, pbarDesc=desc
        )
        self.progressWin.mainPbar.setMaximum(0)
        self.progressWin.show(self.app)

        self.magicPromptsThread = QThread()
        self.magicPromptsWorker = workers.MagicPromptsWorker(
            posData, image, df_points, model, model_segment_kwargs,
            image_origin=image_origin,
            global_image=posData.img_data[posData.frame_i]
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
            partial(self.magicPromptsWorkerFinished, zoom_slice=zoom_slice)
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

    def magicPromptsWorkerFinished(self, output, zoom_slice=None):
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

        img = (
            posData.img_data[posData.frame_i][..., zoom_slice[0], zoom_slice[1]]
        )
        images = [img, img, img, img]
        labels_overlays = [
            posData.lab[..., zoom_slice[0], zoom_slice[1]],
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

        lab = posData.allData_li[posData.frame_i]['labels']
        lab[..., zoom_slice[0], zoom_slice[1]][zoom_out_lab_mask] = (
            zoom_out_lab[zoom_out_lab_mask]
        )

        posData.allData_li[posData.frame_i]['labels'] = lab
        self.get_data()
        self.store_data(autosave=False)
        self.updateAllImages()
