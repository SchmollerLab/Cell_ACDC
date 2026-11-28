"""Loading, saving, reloading and exporting data in the measurements viewer
(ViewerDataIOGuiElements), the loading guards and the unsaved-changes
prompts. Spec: measurements_viewer/__init__.py.
"""
import os
import html
import time
from qtpy.QtCore import Qt, QTimer, QSaveFile, QIODevice, QThread, QEventLoop
from qtpy.QtWidgets import QFileDialog, QApplication
from cellacdc import load, qutils
from cellacdc.shared.widgets import (
    item_views as widgets_item_views,
    message_box as widgets_message_box,
)
from cellacdc import html_utils
from cellacdc import disableWindow
from ..annotation_data import (
    VISITED_COLUMN,
    flag_column_conflicts,
    flag_column_conflicts_text,
)
from ..position_data import acdc_df_file_stamp, analysis_last_frame
from ..viewer_utils import (
    LAST_ANALYZED_CCA,
    acdc_df_snapshot,
    get_channel_from_filepath,
    get_channels_and_segm_endnames,
    get_images_paths,
    load_display_positions,
    normalize_loaded_acdc_df,
    position_name,
    update_last_analyzed_frame,
)
from .. import workers as viewer_workers


class ViewerDataIOGuiElements:
    """Opening folders and files, loading in a worker thread (runInThread),
    refusing re-entrant loads, setData/loadPosition, the
    unanalyzed-frames mask, reloading, saving and exporting the
    acdc_output tables and the unsaved-changes prompts.
    """

    def askAcdcDfChangedOnDisk(self, posData):
        """Ask what to do with a table whose file changed on disk since it
        was loaded. Return 'reload', 'overwrite' or None (cancel)."""
        filepath = getattr(posData, 'acdc_output_csv_path', '')
        posName = getattr(posData, 'pos_foldername', filepath)
        txt = html_utils.paragraph(
            f'The acdc_output file of <b>{posName}</b> was modified on disk '
            '(e.g., saved in the main GUI) after the measurements viewer '
            'loaded it.<br><br>'
            '<b>Reload and save</b> reads the file again, keeps the time '
            'points, custom annotations, visited cells and evaluated phases '
            'of the viewer and saves them (other in-memory changes, such as '
            'processed or combined measurements, are discarded).<br><br>'
            '<b>Overwrite</b> replaces the file with the viewer\'s table '
            '(changes saved by other programs are lost).<br><br>'
            '<i>The previous file is backed up in the "recovery" folder in '
            'both cases.</i>'
        )
        msg = widgets_message_box.myMessageBox(wrapText=False)
        _, overwrite, reload = msg.warning(
            self, 'Table changed on disk', txt, detailsText=filepath or None,
            buttonsTexts=('Cancel', 'Overwrite', 'Reload and save')
        )
        if msg.cancel:
            return None
        if msg.clickedButton == reload:
            return 'reload'
        if msg.clickedButton == overwrite:
            return 'overwrite'
        return None

    def saveAcdcDfs(self, positions):
        # Pending phase evaluations of positions not displayed would
        # otherwise be saved stale (e.g. after renaming an annotation that
        # overrides phases)
        self.flushPendingPhaseEvaluations(positions)
        for posData in positions:
            filepath = getattr(posData, 'acdc_output_csv_path', None)
            if filepath and self.isAcdcDfChangedOnDisk(posData):
                answer = self.askAcdcDfChangedOnDisk(posData)
                if answer is None:
                    return False
                if answer == 'reload':
                    self.reloadAcdcDfFile(posData)
                    if self.data and posData is self.posData():
                        self.refreshAcdcDf()
                    # A changed last analyzed frame queued the re-evaluation
                    # of the automatic phases: save them too (the window may
                    # close right after this save)
                    self.flushPendingPhaseEvaluations([posData])
            try:
                if not filepath:
                    raise OSError('No acdc_output file path for this position.')
                acdc_df = self._standardizeAcdcDf(posData.acdc_df).sort_index()
                acdc_df = acdc_df.copy()
                self.ensureTimepointColumns(acdc_df)
                self.ensureCustomAnnotColumns(acdc_df)
                if VISITED_COLUMN not in acdc_df:
                    acdc_df[VISITED_COLUMN] = False
                payload = acdc_df.to_csv().encode('utf-8')
                # Back up the previous file (restorable from the main GUI)
                load.store_copy_acdc_df(
                    posData, filepath, log_func=self.logger.warning
                )
                output = QSaveFile(filepath)
                if not output.open(QIODevice.WriteOnly):
                    raise OSError(output.errorString())
                if output.write(payload) != len(payload):
                    output.cancelWriting()
                    raise OSError(output.errorString())
                if not output.commit():
                    raise OSError(output.errorString())
            except (OSError, ValueError, TypeError) as err:
                self.logger.error('Could not save "%s": %s', filepath, err)
                self.measurementProcessingWarning(
                    'Saving tables failed',
                    f'Could not save the table of '
                    f'"{getattr(posData, "pos_foldername", filepath)}":\n\n{err}'
                )
                return False
            posData.savedAcdcDf = acdc_df.copy()
            posData.acdcDfFileStamp = acdc_df_file_stamp(filepath)
            self.logger.info(f'Table saved to "{filepath}"')
        if positions:
            self.statusBar().showMessage(
                f'Saved {len(positions)} acdc_output table(s).', 5000
            )
        return True

    def saveTables(self, checked=False):
        """Save the acdc_output tables of the positions with unsaved changes
        (File > Save tables, Ctrl+S). Tables without changes are not written,
        so that files saved by the main GUI in the meantime are kept. Return
        False if saving was refused, cancelled or failed."""
        if not self.isDataLoaded:
            # Qt5 emits `triggered` also for a disabled action
            self.statusBar().showMessage('Load data before saving tables.', 5000)
            return False
        if self._isLoadingData:
            text = 'Wait for the data being loaded before saving the tables.'
            self.logger.warning(text)
            self.statusBar().showMessage(text, 8000)
            return False
        positions = self.unsavedChangesToSave()[0]
        if not positions:
            text = 'Save tables: no unsaved changes to the tables.'
            self.logger.info(text)
            self.statusBar().showMessage(text, 5000)
            return True
        self.logger.info(f'Save tables: {len(positions)} position(s)')
        return self.saveAcdcDfs(positions)

    def askSaveBeforeClosing(self):
        """Prompt for unsaved tables/configuration. Return False to cancel."""
        return self.askSaveChanges('closing')

    def askSaveBeforeLoading(self):
        """Prompt for unsaved tables/configuration before other data (a
        configuration, folder or file) replaces the loaded data. Return False
        to cancel the load (no prompt and True when nothing is loaded)."""
        if not self.isDataLoaded:
            return True
        return self.askSaveChanges('loading other data')

    def unsavedChangesToSave(self):
        """Return the positions with unsaved tables and whether the
        configuration has unsaved changes, after running the deferred phase
        evaluations of every loaded position (`flushPendingPhaseEvaluations`):
        they change tables of positions that are not displayed (e.g.
        "Overrides phases" switched), which would otherwise be neither listed
        nor saved."""
        self.flushPendingPhaseEvaluations(list(self.data or []))
        return self.unsavedAcdcDfPositions(), self.hasUnsavedConfiguration()

    def askSaveChanges(self, action='closing'):
        """Prompt for unsaved tables/configuration before `action` (text of
        the prompt). Return False to cancel.

        The prompt does not block the window: if tables or the configuration
        get changes it did not list while it is open, any answer but Cancel
        shows it again with the new changes. Save tables saves the tables
        changed when it is answered."""
        positions, configChanged = self.unsavedChangesToSave()
        while positions or configChanged:
            choice = self.promptSaveChanges(positions, configChanged, action)
            if choice is None:
                return False
            listed = {id(posData) for posData in positions}
            wasConfigChanged = configChanged
            positions, configChanged = self.unsavedChangesToSave()
            if (any(id(posData) not in listed for posData in positions)
                    or configChanged and not wasConfigChanged):
                self.logger.info(
                    'Changes made while the "Save changes?" prompt was open: '
                    'asking again'
                )
                continue
            saveTables, saveConfig = choice
            if saveTables and not self.saveAcdcDfs(positions):
                return False
            if saveConfig is not None and not self.saveConfiguration(
                    save_as=saveConfig == 'save_as'):
                return False
            return True
        return True

    def promptSaveChanges(self, positions, configChanged, action):
        """Show the "Save changes?" prompt for the unsaved tables of
        `positions` and the configuration (`configChanged`). Return the
        chosen ``(saveTables, saveConfig)`` (`saveConfig`: None, 'save' or
        'save_as') or None if cancelled."""
        hasFile = self.configurationFilepath is not None
        saveConfigText = (
            'Save configuration' if hasFile else 'Save configuration...'
        )
        changes = []
        if positions:
            changes.append(
                f'the <b>measurement tables</b> of {len(positions)} '
                f'position(s)'
            )
        if configChanged:
            changes.append('the <b>configuration</b>')
        txt = html_utils.paragraph(
            f'There are unsaved changes to {" and ".join(changes)}.<br><br>'
            f'Do you want to save them before {action}?<br><br>'
            '<i>Saving the tables overwrites each position\'s acdc_output '
            'file (the previous file is backed up in the "recovery" '
            'folder; files modified on disk since loading are reported '
            'first).</i>'
        )
        details = '\n'.join(
            getattr(posData, 'acdc_output_csv_path', '') for posData in positions
        )
        options = [('Cancel', None), ("Don't save", (False, None))]
        if positions and configChanged:
            options += [
                ('Save tables only', (True, None)),
                (f'{saveConfigText} only', (False, 'save')),
            ]
            if hasFile:
                options.append(('Save both as...', (True, 'save_as')))
            options.append(('Save both', (True, 'save')))
        elif positions:
            options.append(('Save tables', (True, None)))
        else:
            if hasFile:
                options.append(('Save configuration as...', (False, 'save_as')))
            options.append((saveConfigText, (False, 'save')))
        msg = widgets_message_box.myMessageBox(wrapText=False)
        buttons = msg.warning(
            self, 'Save changes?', txt, detailsText=details or None,
            buttonsTexts=[text for text, _ in options]
        )
        if msg.cancel:
            self.logger.info(f'Save changes before {action}: Cancel')
            return None
        texts = dict(zip(buttons, (text for text, _ in options)))
        self.logger.info(
            f'Save changes before {action}: '
            f'"{texts.get(msg.clickedButton, "Cancel")}"'
        )
        choices = dict(zip(buttons, (choice for _, choice in options)))
        return choices.get(msg.clickedButton)

    def askMaskUnanalyzedFrames(self, partial, data):
        """Ask whether the calculations should ignore the frames after the
        last analyzed frame of the `partial` positions (of the loaded
        `data`). Return True (ignore them, default), False (use all frames)
        or None if the user cancels."""
        lines = '<br>'.join(
            f'&nbsp;&nbsp;- {html.escape(self.lastAnalyzedFrameText(posData))}'
            for posData in partial[:5]
        )
        if len(partial) > 5:
            lines += f'<br>&nbsp;&nbsp;- ... ({len(partial)-5} more)'
        txt = html_utils.paragraph(
            f'{len(partial)} of {len(data)} position(s) have <b>not been '
            'analyzed until the end of the video</b> in the main GUI '
            '(last cell cycle annotated frame, or last tracked frame '
            f'without cell cycle annotations):<br><br>{lines}<br><br>'
            'Do you want to <b>ignore the frames after the last analyzed '
            'frame</b> in all calculations (time point and custom '
            'annotation predictors, phases, measurement processing, '
            'combined measurements and conditional formatting)? Their '
            'outputs then stay empty, while images and plots still show '
            'all frames.<br><br>'
            'You can change this later with <code>Annotations --> Ignore '
            'frames after the last analyzed frame</code>; the choice is '
            'saved in the configuration.'
        )
        details = '\n'.join(
            self.lastAnalyzedFrameText(posData) for posData in partial
        )
        msg = widgets_message_box.myMessageBox(wrapText=False)
        cancel, useAll, ignore = msg.question(
            self, 'Video not analyzed until the end', txt,
            detailsText=details,
            buttonsTexts=(
                'Cancel', 'Use all frames', 'Ignore later frames (recommended)'
            )
        )
        if msg.cancel or msg.clickedButton == cancel:
            return None
        return msg.clickedButton != useAll

    def resolveMaskUnanalyzedFrames(self, data, stored=None, default=True):
        """Whether the calculations of the loaded `data` ignore the frames
        after the last analyzed frame: the `stored` choice of a loaded
        configuration (bool), else the answer to `askMaskUnanalyzedFrames`
        when some positions are partially analyzed, else `default`. Return
        ``(mask, asked)``; `mask` is None if the user cancels."""
        partial = self.partiallyAnalyzedPositions(data)
        if isinstance(stored, bool):
            mask, asked = stored, False
        elif partial:
            mask, asked = self.askMaskUnanalyzedFrames(partial, data), True
            if mask is None:
                self.logger.info('Loading cancelled (unanalyzed frames prompt)')
                return None, asked
        else:
            mask, asked = bool(default), False
        for posData in partial:
            self.logger.info(
                f'{self.lastAnalyzedFrameText(posData)}: later frames '
                f'{"ignored" if mask else "used"} by the calculations'
            )
        return mask, asked

    def loadingInProgress(self):
        """Return True, and tell the user (status bar and log), while other
        data is being loaded (`_isLoadingData`), to refuse a second load:
        the dialogs of a load do not block the window, so a load could
        otherwise start while another one is waiting for an answer.
        Loading is also refused while the "Save changes before closing?"
        prompt is open (`_isClosingPrompt`): it would save or close the
        replaced data."""
        if self._isClosingPrompt:
            text = 'Answer the "Save changes before closing?" prompt first.'
        elif self._isLoadingData:
            text = (
                'Wait for the data being loaded (or answer its dialog) first.'
            )
        else:
            return False
        self.logger.warning(f'Loading other data refused. {text}')
        self.statusBar().showMessage(text, 8000)
        return True

    def refuseLoadingWhileProcessing(self):
        """Return True, and tell the user, while measurement processing
        runs: its results would be merged into (and its step recorded for)
        the data being replaced."""
        if self.measurementProcessingThread is None:
            return False
        self.configurationError(
            'load', 'Wait for measurement processing to finish.'
        )
        return True

    def refuseProcessingWhileLoading(self):
        """Return True, and tell the user, while data is being loaded (see
        `refuseLoadingWhileProcessing`) or the "Save changes before
        closing?" prompt is open."""
        if self._isClosingPrompt:
            text = 'Answer the "Save changes before closing?" prompt first.'
        elif self._isLoadingData:
            text = (
                'Wait for the data being loaded (or answer its dialog) first.'
            )
        else:
            return False
        self.logger.warning(f'Process measurements refused. {text}')
        self.statusBar().showMessage(text, 8000)
        return True

    def openRecentFile(self, path):
        if self.loadingInProgress():
            return
        self.logger.info(f'Opening recent folder: {path}')
        self.addToRecentPaths(path)
        self.openFolder(folderpath=path)

    def openFolder(self, checked=False, folderpath=None):
        if self.loadingInProgress():
            return
        if not folderpath:
            folderpath = QFileDialog.getExistingDirectory(
                self, 'Select experiment, Position or Images folder', 
                self.getMostRecentPath()
            )
        if not folderpath:
            return
        
        self.logger.info(f'Opening folder "{folderpath}"...')
        self.addToRecentPaths(folderpath)
        self.addPathToOpenRecentMenu(folderpath)
        
        images_paths = get_images_paths(folderpath)
        if not images_paths:
            self.warnLoading(
                f'No Position or Images folders found in "{folderpath}".'
            )
            return
        
        self.loadImagesPaths(images_paths)

    def openFile(self, checked=False, file_path=None):
        if self.loadingInProgress():
            return
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 'Select image file from an Images folder', 
                self.getMostRecentPath(), 
                'Images (*.tif *.tiff *.npz *.npy *.h5);;All Files (*)'
            )
        if not file_path:
            return
        
        self.logger.info(f'Opening file "{file_path}"...')
        images_path = os.path.dirname(file_path)
        if os.path.basename(images_path) != 'Images':
            self.warnLoading(
                f'The file "{file_path}" is not in an "Images" folder.\n\n'
                'Create the Cell-ACDC data structure first (e.g., by opening '
                'the file in the main GUI).'
            )
            return
        
        self.addToRecentPaths(images_path)
        self.addPathToOpenRecentMenu(images_path)
        channel = get_channel_from_filepath(file_path)
        self.loadImagesPaths([images_path], channel=channel)

    def loadImagesPaths(self, images_paths, channel=None):
        if self.loadingInProgress():
            return
        self._isLoadingData = True
        try:
            self._loadImagesPaths(images_paths, channel=channel)
        finally:
            self._isLoadingData = False

    def _loadImagesPaths(self, images_paths, channel=None):
        if self.refuseLoadingWhileProcessing():
            return
        try:
            chNames, segm_endnames = get_channels_and_segm_endnames(
                images_paths[0]
            )
        except Exception as err:
            self.warnLoading(str(err))
            return
        
        if not segm_endnames:
            self.warnLoading(
                f'No segmentation files found in "{images_paths[0]}".'
            )
            return
        
        if channel is None:
            if not chNames:
                self.warnLoading(
                    f'No channel images found in "{images_paths[0]}".'
                )
                return
            channel = self.selectItem(
                'Select channel', 'Select channel to display:', chNames
            )
        if channel is None:
            return
        
        segm_endname = self.selectItem(
            'Select segmentation file', 'Select segmentation file to load:', 
            segm_endnames
        )
        if segm_endname is None:
            return

        if not self.askSaveBeforeLoading():
            return
        # Processing may have been started from the non-blocking dialogs
        if self.refuseLoadingWhileProcessing():
            return

        self.logger.info(
            f'Loading channel "{channel}" and segmentation '
            f'"{segm_endname}" of {len(images_paths)} position(s): '
            + ', '.join(f'"{path}"' for path in images_paths)
        )
        # In a worker thread: the window stays responsive (Positions that
        # cannot be loaded are skipped and reported in the terminal)
        data = self.runInThread(
            load_display_positions, images_paths, channel,
            segm_endname=segm_endname, reportProgress=True,
            text=f'Loading {len(images_paths)} position(s)...'
        )

        if not data:
            self.warnLoading(
                'None of the selected Positions could be loaded '
                '(see terminal for details).'
            )
            return

        # The annotations persist across loading: `setData` would turn such
        # columns of the new tables into 0/1 (`customAnnotColumnConflict`)
        conflicts = flag_column_conflicts(
            self.customAnnotNames(),
            [getattr(posData, 'acdc_df', None) for posData in data]
        )
        if conflicts:
            text = flag_column_conflicts_text(conflicts)
            self.logger.warning('Loading refused. ' + text.replace('\n', ' '))
            self.warnLoading(
                f'{text}\n\nRename or remove these custom annotations, then '
                'open the data again.'
            )
            return

        # Before the predictions of `setData`
        mask, _ = self.resolveMaskUnanalyzedFrames(
            data, default=self.maskUnanalyzedFrames
        )
        if mask is None:
            return

        # The recorded measurement steps stay: their saved outputs are used
        # as existing values (as in `loadConfiguration`)
        outputs = self.measurementStepOutputs(self.configurationSteps())
        # The record of the saved configuration, before blanking changes it
        saved = self.savedConfiguration
        savedStepsLastFrames = dict(
            saved.get('measurement_steps_last_frames', {})
            if isinstance(saved, dict) else self.measurementStepsLastFrames
        )
        if outputs:
            lastFrames = self.blankSavedStepOutputs(
                data, outputs, mask, self.measurementStepsLastFrames
            )
            for posData, last in zip(data, lastFrames):
                current = analysis_last_frame(posData, mask)
                if last is not None and (current is None or last < current):
                    self.logger.info(
                        f'{position_name(posData.images_path)}: the '
                        'measurement steps are calculated only up to frame '
                        f'n. {last+1}: load the configuration again and '
                        'choose "Calculate" to update them.'
                    )

        self.isSegm3D = None
        self.setMaskUnanalyzedFrames(mask, update=False)
        with self.deferPredictorPredictions():
            # The session prediction state belongs to the replaced data
            # (e.g. reopened after "Don't save"): rebuild it from the
            # visited column of the loaded tables.
            self.resetAutoPredictionTracking()
            self.setData(data)
            # The annotations and time point identifiers persist across
            # loading: saved assignments and time points of visited cells
            # must survive auto-prediction (as in `loadConfiguration`).
            self.seedVisitedCustomAnnotPredictions(
                self.customAnnotNames(), data
            )
            self.seedVisitedTimepointPredictions(data=data)
        self.markConfigurationSaved()
        if (self.measurementStepsLastFrames != savedStepsLastFrames
                and isinstance(self.savedConfiguration, dict)):
            # Existing values blanked up to other frames: the new record (it
            # protects the saved step outputs) is not saved yet (as in
            # `loadConfiguration`)
            self.savedConfiguration['measurement_steps_last_frames'] = (
                savedStepsLastFrames
            )
        self.logger.info(f'Loaded {len(data)} position(s)')
        self.setStatusBarLabel()

    def runInThread(
            self, func, *args, text='Working...', reportProgress=False,
            **kwargs
        ):
        """Run `func(*args, **kwargs)` in a worker `QThread` and return its
        result, or re-raise the exception it raised.

        A local event loop keeps the window repainting meanwhile (a long
        task on the GUI thread makes it "Not Responding", and Windows then
        shows a frozen copy of it), with its input disabled and a wait
        cursor. `text` is logged and shown in the status bar, followed by
        the progress texts that `func` reports through its ``progress``
        keyword argument (passed with `reportProgress`) and by the elapsed
        time. `func` must not create or touch Qt widgets nor show dialogs.

        The other windows of the application (launcher, main GUI, other
        viewers) are disabled meanwhile too: a dialog opened from them would
        run its own local event loop inside this one, and this call could
        return (re-enabling the window) only once that dialog closed.
        """
        worker = viewer_workers._ThreadTaskWorker(func, args, kwargs, reportProgress)
        thread = QThread()
        worker.moveToThread(thread)
        loop = QEventLoop()
        thread.started.connect(worker.run)
        # Queued to this (GUI) thread: received only while `loop` runs
        thread.finished.connect(loop.quit)
        worker.sigProgress.connect(self.showThreadTaskProgress)
        wasEnabled = self.isEnabled()
        self.setEnabled(False)
        otherWindows = [
            window for window in QApplication.topLevelWidgets()
            if window is not self and window.isVisible()
            and window.isEnabled() and not self.isAncestorOf(window)
        ]
        for window in otherWindows:
            window.setEnabled(False)
        self.setCursor(Qt.WaitCursor)
        self.logger.info(text)
        self.statusBar().showMessage(text)
        t0 = time.perf_counter()
        try:
            thread.start()
            loop.exec_()
        finally:
            thread.wait()
            thread.deleteLater()
            # `worker` moved back to this thread at the end of `run`
            worker.deleteLater()
            self.unsetCursor()
            for window in otherWindows:
                try:
                    window.setEnabled(True)
                except RuntimeError:
                    # Deleted meanwhile
                    pass
            if wasEnabled:
                self.setEnabled(True)
            self.statusBar().clearMessage()
        elapsed = time.perf_counter() - t0
        action = text.rstrip('.').strip()
        if worker.error is not None:
            self.logger.info(f'{action} failed after {elapsed:.1f} s')
            raise worker.error
        self.logger.info(f'{action} done in {elapsed:.1f} s')
        return worker.result

    def showThreadTaskProgress(self, text):
        """Show (and log) a progress text of a `runInThread` task."""
        self.logger.info(text)
        self.statusBar().showMessage(text)

    def selectItem(self, title, text, items):
        """Return the item chosen in a single-choice list (None if
        cancelled or `items` is empty: callers with possibly empty `items`
        report it themselves). OK with nothing selected (Ctrl+Click on the
        selected item) warns and keeps the list open: an empty answer would
        end the calling flow without any message (e.g., the base layer of a
        premade configuration)."""
        if not items:
            return None
        if len(items) == 1:
            return items[0]
        
        win = widgets_item_views.QDialogListbox(
            title, text, items, multiSelection=False, parent=self,
            allowEmptySelection=False
        )
        win.exec_()
        if win.cancel or not win.selectedItemsText:
            return
        return win.selectedItemsText[0]

    def warnLoading(self, text):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'Loading data failed', text)

    def setData(self, data, pos_i=0, frame_i=None, ID=None):
        if self.data:
            # Turn the tools off while the previous data is still loaded,
            # so that their slots (e.g. `overlay_cb`) can tear down normally
            self.uncheckViewerTools()
        self.data = list(data) if data else []
        self.prunePendingPhaseIDs()
        hasData = len(self.data) > 0
        # `pos_i` of previously loaded data might be out of range
        pos_i = min(max(pos_i, 0), len(self.data)-1) if hasData else 0
        self.pos_i = pos_i
        self.centralWidget().setEnabled(hasData)
        self.setEnabledFileToolbar(hasData)
        
        self.posCombobox.blockSignals(True)
        self.posCombobox.clear()
        for posData in self.data:
            self.posCombobox.addItem(
                getattr(posData, 'pos_foldername', str(posData))
            )
        self.posCombobox.blockSignals(False)
        
        if not hasData:
            isMultiPos = False
            self.posLabel.setVisible(isMultiPos)
            self.posCombobox.setVisible(isMultiPos)
            self.setStatusBarLabel(log=False)
            return
        
        relPath = getattr(self.data[0], 'relPath', '')
        self.setWindowTitle(f'Cell-ACDC - Measurements viewer - {relPath}')
        self.pos_i = pos_i
        self.setPosDataDefaults()
        self.initLut()
        self.initViewerTools()
        self.loadPosition(pos_i, frame_i=frame_i, ID=ID)
        # As the main GUI (`loadingDataCompleted`): show the custom
        # annotations of every button, not only of the active one
        # (unchecking the eye button of the toolbar shows only the active
        # one and those whose "Hide annotations" option is off)
        self.viewAllCustomAnnotAction.setChecked(True)
        self.initViewerToolsAfterLoading()
        self.setEnabledFileToolbar(hasData)
        self.resizeBottomLayout()
        # On the first pass of the event loop, i.e. once the load finished:
        # the synchronous work after `setData` (e.g. `loadOverlayData` in
        # `loadConfiguration`) would otherwise delay the blink's timers
        # until it is over, hiding the blink.
        QTimer.singleShot(0, self.blinkCellIdControls)

    def blinkCellIdControls(self):
        """Blink the Cell ID spinbox and scrollbar after loading data, to
        show where the selected (highlighted) cell is changed."""
        if not self.data:
            return
        blinker = qutils.QControlBlink(
            [self.idLabel, self.idSpinBox, self.idScrollBar], qparent=self
        )
        blinker.start()
        self.statusBar().showMessage(
            'Select the cell to plot with the "Cell ID" controls (bottom) '
            'or click on a cell in the image.', 10000
        )

    def resizeBottomLayout(self):
        # As in `gui.guiWin.resizeGui`: fit the bottom layout to its 
        # content and then restore the height saved by the user
        self.bottomScrollArea._resizeVertical()
        self.setBottomLayoutHeight()

    def loadPosition(self, pos_i, frame_i=None, ID=None):
        self.pos_i = pos_i
        posData = self.posData()
        self._lab = None
        self._labKey = None
        
        if self.isSegm3D is None:
            self.isSegm3D = posData.segm_data.ndim == 4
        
        if frame_i is None:
            frame_i = max(getattr(posData, 'frame_i', 0), 0)
        self.frame_i = min(frame_i, posData.SizeT-1)
        posData.frame_i = self.frame_i
        
        self.logger.info(
            f'Loading position {pos_i+1}/{len(self.data)} '
            f'("{getattr(posData, "pos_foldername", "")}"), '
            f'frame n. {self.frame_i+1}...'
        )
        self.posCombobox.blockSignals(True)
        self.posCombobox.setCurrentIndex(pos_i)
        self.posCombobox.blockSignals(False)
        isMultiPos = len(self.data) > 1
        self.posLabel.setVisible(isMultiPos)
        self.posCombobox.setVisible(isMultiPos)

        # Initialize the position before the navigation widgets, whose
        # range changes must never act on a partially loaded position
        allIDs = self.initPosAnnotData(posData)
        self._IDframesCache = {}
        self.refreshAcdcDf(plot=False)
        # Cells of this position whose phase evaluation was postponed
        self.schedulePendingPhaseAutoApply()
        self.initFramesNavigation()
        self.updateLastAnalyzedFrameLabel()
        self.initZsliceNavigation()
        self.initIDsNavigation(ID=ID)
        self.updateFramesRange()
        
        self.get_data()
        self.initContoursImage()
        self.gui_createAxesTextAnnotItems(allIDs)
        self.updateImages()
        self.ax1.autoRange()
        self.zoomToSelectedID()
        self._timecourseXRangeKey = None
        self._comparisonRangeKey = None
        self.plotMeasurements()

        if self.rescalePlotCheckbox.isChecked():
            self.autoscaleMeasurementsPlot()
        # Logged at the end of a load (Open folder, configuration), which
        # can still change it (e.g., the configuration file)
        self.setStatusBarLabel(log=not self._isLoadingData)

    def setMaskUnanalyzedFrames(self, mask, update=True):
        """Set the "Ignore frames after the last analyzed frame" option
        (and its menu action, without emitting `toggled`). With `update`,
        the loaded data are updated if the option changed
        (`analysisFramesChanged`)."""
        mask = bool(mask)
        changed = mask != self.maskUnanalyzedFrames
        self.maskUnanalyzedFrames = mask
        action = getattr(self, 'maskUnanalyzedFramesAction', None)
        if action is not None and action.isChecked() != mask:
            action.blockSignals(True)
            action.setChecked(mask)
            action.blockSignals(False)
        if update and changed:
            self.analysisFramesChanged(
                self.partiallyAnalyzedPositions(self.data or [])
            )

    def maskUnanalyzedFramesToggled(self, checked):
        if bool(checked) == self.maskUnanalyzedFrames:
            return
        if self._isLoadingData or self.measurementProcessingThread is not None:
            # The load sets the option from its prompt or configuration;
            # running processing uses the frames of when it started
            self.setMaskUnanalyzedFrames(self.maskUnanalyzedFrames)
            self.statusBar().showMessage(
                'Wait for the data being loaded (or answer its dialog) first.'
                if self._isLoadingData else
                'Wait for measurement processing to finish first.',
                8000
            )
            return
        self.logger.info(
            'Ignore frames after the last analyzed frame '
            f'{"on" if checked else "off"}'
        )
        self.setMaskUnanalyzedFrames(checked)
        partial = self.partiallyAnalyzedPositions(self.data or [])
        if not partial:
            return
        self.statusBar().showMessage(
            ('Frames after the last analyzed frame are now ignored'
             if checked else 'All frames are now used')
            + ' by new calculations. Re-evaluate the predictors (and '
            'phases), or reload the configuration to recalculate the '
            'measurement steps.', 12000
        )

    def analysisFramesChanged(self, positions=None, replot=True):
        """Update the loaded `positions` (default: all) after the frames
        used by their calculations changed (option toggled, or table
        reloaded with a different last analyzed frame): the phases of all
        their cells are re-evaluated when they are evaluated automatically
        (the other outputs are not recalculated: their time points,
        annotations and measurement steps may have manual edits or
        parameters to review), the label and the plot are updated."""
        if not self.data:
            return
        if positions is None:
            positions = self.data
        positions = list(positions)
        for posData in positions:
            # The position's table (just reloaded, while `self.acdc_df` may
            # still be the previous one until `refreshAcdcDf`)
            acdc_df = getattr(posData, 'acdc_df', None)
            if acdc_df is None or acdc_df.empty:
                continue
            acdc_df = self._standardizeAcdcDf(acdc_df)
            if getattr(posData, 'phaseAutoEvaluatedIDs', None) is not None:
                posData.phaseAutoEvaluatedIDs.clear()
            self.schedulePhaseAutoApply(
                set(acdc_df.index.get_level_values('Cell_ID')), posData
            )
        self.updateLastAnalyzedFrameLabel()
        if replot and self.isDataLoaded and any(
                posData is self.posData() for posData in positions):
            self.plotMeasurements()

    def updateLastAnalyzedFrameLabel(self):
        """Show the last analyzed frame of the displayed position next to
        the frames scrollbar (as the main GUI's last tracked frame), when
        it is not analyzed until its last frame."""
        label = getattr(self, 'lastAnalyzedFrameLabel', None)
        if label is None:
            return
        posData = self.posData() if self.data else None
        if posData is None or analysis_last_frame(posData) is None:
            label.setText('')
            return
        kind = getattr(posData, 'lastAnalyzedKind', None)
        kind = 'cc annot.' if kind == LAST_ANALYZED_CCA else 'tracked'
        text = f'Last {kind} frame n. = {posData.lastAnalyzedFrame+1}'
        if self.maskUnanalyzedFrames:
            text = f'{text} (later frames ignored)'
        else:
            text = f'{text} (all frames used)'
        label.setText(text)

    def reloadAcdcDfFile(self, posData=None):
        if posData is None:
            posData = self.posData()
        
        acdc_df_path = getattr(posData, 'acdc_output_csv_path', None)
        if not acdc_df_path or not os.path.exists(acdc_df_path):
            self.logger.warning(
                f'acdc_output CSV file not found: {acdc_df_path}'
            )
            self.statusBar().showMessage(
                f'acdc_output CSV file not found: {acdc_df_path}', 8000
            )
            return
        
        previous = getattr(posData, 'acdc_df', None)
        previousLast = self.analysisLastFrame(posData)
        posData.loadAcdcDf(acdc_df_path)
        # Results of measurement processing started before are stale.
        posData.acdcDfReloadCount = getattr(posData, 'acdcDfReloadCount', 0) + 1
        normalize_loaded_acdc_df(posData)
        # More frames may have been annotated in the main GUI meanwhile
        update_last_analyzed_frame(posData)
        posData.savedAcdcDf = acdc_df_snapshot(getattr(posData, 'acdc_df', None))
        posData.acdcDfFileStamp = acdc_df_file_stamp(acdc_df_path)
        self.restoreTimepointColumns(posData, previous)
        last = self.analysisLastFrame(posData)
        # Step outputs are empty after the frame they were calculated up to
        # (saved NaN is read back as 0 from tables with cell cycle columns)
        # and are ignored after the current last analyzed frame
        outputs = self.measurementStepOutputs(self.configurationSteps())
        blankFrom, = self.blankSavedStepOutputs(
            [posData], outputs, self.maskUnanalyzedFrames,
            self.measurementStepsLastFrames
        )
        if last != previousLast:
            text = (
                f'{position_name(getattr(posData, "images_path", ""))}: '
                + ('new calculations use all frames'
                   if last is None else
                   f'new calculations use frames 1-{last+1}')
                + ' (last analyzed frame changed).'
            )
            if outputs and blankFrom is not None and (
                    last is None or blankFrom < last):
                text += (
                    ' The measurement steps are calculated only up to frame '
                    f'n. {blankFrom+1}: load the configuration again and '
                    'choose "Calculate" to update them.'
                )
            self.logger.info(text)
            self.statusBar().showMessage(text, 12000)
            # The callers refresh (and replot) the displayed table
            self.analysisFramesChanged([posData], replot=False)
        else:
            # The last analyzed frame (or its kind) shown in the label may
            # have changed (e.g. with the option off)
            self.updateLastAnalyzedFrameLabel()

    @disableWindow
    def reloadButtonClicked(self, checked=False):
        self.logger.info(
            'Reloading table "'
            f'{getattr(self.posData(), "acdc_output_csv_path", "")}"...'
        )
        self.reloadAcdcDfFile()
        self.refreshAcdcDf()

    def exportMeasurements(self, checked=False):
        if self.acdc_df is None or self.acdc_df.empty:
            self.measurementProcessingWarning('No measurements', 'Load measurements first.')
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, 'Export measurements', 'processed_measurements.csv', 'CSV (*.csv)'
        )
        if not filepath:
            return
        if self.hasPhaseTranslation():
            msg = widgets_message_box.myMessageBox(wrapText=False)
            cancel, skip, apply = msg.question(
                self, 'Evaluate phases before export?',
                'A phase translator (phase rules or custom annotations '
                'overriding phases) is configured. Evaluate all cells of the '
                'current position before exporting?',
                buttonsTexts=('Cancel', 'Export existing values', 'Evaluate and export')
            )
            if msg.cancel or msg.clickedButton == cancel:
                return
            if msg.clickedButton == apply and not self.applyPhaseTranslator():
                return
        try:
            export_df = self.acdc_df.copy()
            self.ensureTimepointColumns(export_df)
            self.ensureCustomAnnotColumns(export_df)
            self.logger.info(f'Exporting measurements to "{filepath}"...')
            export_df.to_csv(filepath)
        except (OSError, ValueError) as error:
            self.measurementProcessingWarning('Export failed', str(error))
            return
        self.logger.info(f'Measurements exported to "{filepath}"')
        # Temporary: `statusBarLabel` shows what is loaded
        self.statusBar().showMessage(
            f'Measurements exported to "{filepath}".', 8000
        )

    def refreshAcdcDf(self, plot=True):
        self.acdc_df = self.getAcdcDf()
        self._IDframesCache = {}
        self.initLineageTree()
        self.updateColumnsList()
        if not plot:
            return
        self.initIDsNavigation(ID=self.selectedID)
        self.updateFramesRange()
        self.updateImages()
        self.plotMeasurements()
