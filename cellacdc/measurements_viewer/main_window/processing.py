"""Process measurements and Combine measurements in the measurements viewer
(ViewerProcessingGuiElements). Spec: measurements_viewer/__init__.py.
"""
import copy
import traceback
from functools import partial
import numpy as np
import pandas as pd
from qtpy.QtCore import QThread, QMutex, QWaitCondition
from cellacdc import widgets
from cellacdc import apps
from cellacdc import workers
from cellacdc import apps_measurements_viewer
from cellacdc.measurements_viewer_utils import float_if_bool
from .. import viewer_utils


class ViewerProcessingGuiElements:
    """Process measurements (recipe dialog, worker, recorded steps) and
    Combine measurements, with their warnings.
    """

    def measurementProcessingWarning(
            self, title, text, details=None, parent=None
        ):
        """Log and show a warning, parented to the viewer or to `parent`
        (a modal dialog must parent it, or Qt blocks its buttons)."""
        if details:
            self.logger.warning('%s: %s\n%s', title, text, details)
        else:
            self.logger.warning('%s: %s', title, text)
        msg = widgets.myMessageBox(wrapText=False)
        if details:
            msg.setDetailedText(details, wrap=False)
        msg.warning(self if parent is None else parent, title, text)

    def combineMeasurements(self, checked=False):
        if self.measurementProcessingThread is not None:
            self.measurementProcessingWarning(
                'Measurement processing running',
                'Wait for measurement processing to finish before combining measurements.'
            )
            return
        if self.acdc_df is None or self.acdc_df.empty:
            self.measurementProcessingWarning('No measurements', 'Load measurements first.')
            return
        posData = self.posData()
        source = self.acdc_df
        # Equations (and the dialog's test) see only the analyzed frames;
        # `source` stays the full table (identity check when applying)
        self.logger.info('Opening "Combine measurements" dialog...')
        dialog = apps.CombineMetricsEquationDialog(
            posData.chNames, posData.SizeZ > 1, self.isSegm3D,
            parent=self, closeOnOk=False,
            acdc_df=self.analysisDf(source, posData).copy(),
            saveToConfig=False, allowCellAggregations=True,
            limitToLoadedColumns=True
        )
        # `SizeT` of the equations, as when the step is recalculated
        # (`replayMeasurementSteps`)
        dialog.SizeT = self.analysisSizeT(posData)
        dialog.setToolTip(
            'Results are added to the current position in\n'
            'memory. Use Export measurements to save them.'
        )
        dialog.sigOk.connect(
            partial(
                self.applyCombinedMeasurements, posData=posData,
                source=source, lastFrame=self.analysisLastFrame(posData)
            )
        )
        dialog.exec_()
        dialog.deleteLater()

    def applyCombinedMeasurements(
            self, dialog, *, posData, source, lastFrame=None
        ):
        dialog.cancel = True
        if (
            self.measurementProcessingThread is not None
            or not any(pos is posData for pos in self.data)
            or self.posData() is not posData
            or self.acdc_df is not source
            # Frames used by the calculations changed (option toggled)
            or self.analysisLastFrame(posData) != lastFrame
        ):
            self.measurementProcessingWarning(
                'Measurements changed',
                'The source table changed. Reopen Combine measurements and try again.'
            )
            return
        equations, _ = dialog.getEquationsDict()
        # The table as it is now: time points, annotations, etc. may have
        # been edited in place while the (non-blocking) dialog was open
        equationSource = self.analysisDf(source, posData).copy()
        # Index levels too: a column named like one (e.g. frame_i) would be
        # read instead of the index (`mask_frames`) and break `reset_index`
        collisions = set(equations).intersection({
            *equationSource.columns, *source.index.names, *viewer_utils.INDEX_COLUMNS
        })
        if collisions:
            self.measurementProcessingWarning(
                'Invalid measurement name',
                'Choose a name that does not overwrite existing columns: '
                + ', '.join(sorted(collisions))
            )
            return
        conflicts = self.recordedOutputConflicts(equations)
        if conflicts:
            self.recordedOutputConflictWarning(conflicts)
            return
        _, conflicts = self.measurementStepReplacementOrder(
            {'equations': equations}
        )
        if conflicts:
            self.recordedOrderConflictWarning(conflicts)
            return
        self.logger.info(
            'Combining measurements: ' + ', '.join(
                f'{name} = {expression}'
                for name, expression in equations.items()
            )
        )
        try:
            result = source.copy()
            for name, expression in equations.items():
                if equationSource.empty and not source.empty:
                    # No analyzed frame: empty outputs
                    result[name] = np.nan
                    continue
                values = dialog.evaluateEquation(equationSource, expression)
                if not isinstance(values, pd.Series):
                    raise ValueError('The equation must return one value per row.')
                if len(equationSource) < len(source):
                    # NaN on the frames not analyzed (aligned on the
                    # index): keep bool results numeric
                    values = float_if_bool(values)
                result[name] = values
        except Exception as err:
            self.logger.exception('Combining measurements failed')
            self.measurementProcessingWarning('Combining measurements failed', str(err))
            return
        posData.acdc_df = result
        self.acdc_df = result
        self.recordStepsLastFrame(
            self.measurementStepsLastFrames, posData, lastFrame
        )
        note = self.recordMeasurementStep(
            self.combinedMeasurementSteps, {'equations': equations}
        )
        self.updateColumnsList()
        self.plotMeasurements()
        dialog.cancel = False
        dialog.close()
        self.logger.info(
            f'Combined measurements added: {", ".join(equations)}'
        )
        self.statusBar().showMessage(
            'Combined measurements added. Use Export measurements to save them.'
            + (f' {note}' if note else ''),
            10000 if note else 5000
        )

    def processMeasurements(self, checked=False):
        if self.measurementProcessingThread is not None:
            return
        if self.refuseProcessingWhileLoading():
            return
        if self.acdc_df is None or self.acdc_df.empty:
            self.measurementProcessingWarning('No measurements', 'Load measurements first.')
            return
        columns = [
            col for col in self.acdc_df.select_dtypes(include='number').columns
            if col not in viewer_utils.INDEX_COLUMNS
        ]
        if not columns:
            self.measurementProcessingWarning(
                'No measurements', 'The table has no numeric measurements to process.'
            )
            return
        dialog = apps_measurements_viewer.MeasurementProcessingRecipeDialog(
            columns, parent=self, acdc_df=self.acdc_df,
            allChNames=self.posData().chNames
        )
        self.logger.info('Opening "Process measurements" dialog...')
        posData = self.posData()
        dialog.exec_()
        # It keeps the table alive: delete it, its result stays readable
        dialog.deleteLater()
        if dialog.cancel:
            self.logger.info('Process measurements cancelled')
            return
        # The dialog does not block the window: other data may have been
        # loaded (or be loading) meanwhile
        if self.refuseProcessingWhileLoading():
            return
        if not self.data or self.posData() is not posData:
            self.measurementProcessingWarning(
                'Data changed',
                'Other data or another position was loaded while the '
                'dialog was open: start "Process measurements" again.'
            )
            return
        # E.g. started from a second recipe dialog meanwhile
        if self.measurementProcessingThread is not None:
            self.measurementProcessingWarning(
                'Measurement processing running',
                'Wait for measurement processing to finish, then start '
                '"Process measurements" again.'
            )
            return
        # E.g. dropped by Reload table meanwhile
        missing = [
            col for col in dialog.selectedChannels
            if self.acdc_df is None or col not in self.acdc_df.columns
        ]
        if missing:
            self.measurementProcessingWarning(
                'Measurements changed',
                'These measurements are no longer in the table: '
                + ', '.join(missing)
                + '. Start "Process measurements" again.'
            )
            return

        conflicts = self.recordedOutputConflicts(dialog.saveNames)
        if conflicts:
            self.recordedOutputConflictWarning(conflicts)
            return
        _, conflicts = self.measurementStepReplacementOrder({
            'columns': list(dialog.selectedChannels),
            'output_columns': list(dialog.saveNames)
        })
        if conflicts:
            self.recordedOrderConflictWarning(conflicts)
            return
        self.startMeasurementsProcessing(
            dialog.selectedChannels, dialog.selectedRecipe, dialog.saveNames
        )

    def startMeasurementsProcessing(self, columns, recipe, saveNames):
        if self.measurementProcessingThread is not None:
            # One job at a time: its state is kept on the window
            self.logger.warning(
                'Measurement processing is already running: '
                f'{", ".join(saveNames)} not started.'
            )
            return
        posData = self.posData()
        self.logger.info(
            f'Processing measurements {", ".join(columns)} '
            '(recipe: ' + ', '.join(
                str(step.get('method') if isinstance(step, dict) else step)
                for step in recipe
            ) + ') '
            f'into {", ".join(saveNames)}...'
        )
        self._pendingMeasurementStep = copy.deepcopy({
            'columns': list(columns), 'recipe': recipe,
            'output_columns': list(saveNames)
        })
        # Results are merged into the position's table at completion unless
        # the table was reloaded from file or the input columns changed.
        # Only the analyzed frames are processed (NaN on the later ones);
        # the results are stale if they change meanwhile.
        lastFrame = self.analysisLastFrame(posData)
        self._measurementProcessingTarget = (
            posData, getattr(posData, 'acdcDfReloadCount', 0),
            self.acdc_df[list(columns)].copy(), list(saveNames), lastFrame
        )
        self.measurementProcessingThread = QThread(self)
        self.measurementProcessingWorker = workers.MeasurementProcessingWorkerGUI(
            QMutex(), QWaitCondition()
        )
        worker = self.measurementProcessingWorker
        worker.setupJob(
            df=self.acdc_df.copy(),
            cols=columns,
            recipe=recipe,
            new_colnames=saveNames,
            max_frame_i=lastFrame
        )
        worker.moveToThread(self.measurementProcessingThread)
        self.measurementProcessingThread.started.connect(worker.runJob)
        worker.sigDone.connect(self.measurementProcessingDone)
        worker.signals.critical.connect(self.measurementProcessingFailed)
        worker.signals.finished.connect(self.measurementProcessingThread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        self.measurementProcessingThread.finished.connect(self.measurementProcessingFinished)
        self.processMeasurementsButton.setEnabled(False)
        self.combineMeasurementsButton.setEnabled(False)
        self.refreshButton.setEnabled(False)
        self.measurementProcessingThread.start()

    def measurementProcessingDone(self, processed):
        posData, _, inputs, outputs, _ = self._measurementProcessingTarget
        current = self.measurementProcessingTargetTable(posData)
        stale = current is None
        if not stale:
            columns = list(inputs.columns)
            stale = (
                not set(columns).issubset(current.columns)
                or not current[columns].sort_index().equals(inputs.sort_index())
            )
        if stale:
            self.measurementProcessingWarning(
                'Measurements changed',
                'The source table changed during processing. Results were not applied.'
            )
            return
        collisions = [col for col in outputs if col in current.columns]
        if collisions:
            # e.g. added meanwhile by a predictor or custom annotation
            self.measurementProcessingWarning(
                'Invalid measurement name',
                'Columns with the output names were added during processing. '
                'Results were not applied: ' + ', '.join(collisions)
            )
            return

        # Merge only the new columns, so that edits made meanwhile to the
        # other columns (time points, annotations, visited cells) are kept.
        for col in outputs:
            current[col] = processed[col].reindex(current.index)
        posData.acdc_df = current
        self.recordStepsLastFrame(
            self.measurementStepsLastFrames, posData,
            self._measurementProcessingTarget[4]
        )
        self.logger.info(
            f'Measurement processing done: added {", ".join(outputs)}'
        )
        note = self.recordMeasurementStep(
            self.measurementProcessingSteps, self._pendingMeasurementStep
        )
        if note:
            self.statusBar().showMessage(note, 10000)
        if self.posData() is posData:
            self.acdc_df = current
            self.updateColumnsList()
            self.plotMeasurements()

    def measurementProcessingFailed(self, out):
        worker, error = out
        details = ''.join(traceback.format_exception(
            type(error), error, error.__traceback__
        ))
        self.measurementProcessingWarning(
            'Measurement processing failed', str(error), details=details
        )

    def recordedOrderConflictWarning(self, conflicts):
        self.measurementProcessingWarning(
            'Invalid measurement step',
            'This step replaces a recorded step of the configuration whose '
            'outputs are read by other steps, but it reads outputs of steps '
            'recorded later, so the configuration could not be recalculated '
            'in order. Choose other output names or inputs: '
            + ', '.join(conflicts)
        )

    def recordedOutputConflictWarning(self, conflicts):
        self.measurementProcessingWarning(
            'Invalid measurement name',
            'These output names are already written by a different measurement '
            'step of the configuration (each output must come from exactly one '
            'step). Choose other names or re-apply that step with all of its '
            'outputs: ' + ', '.join(conflicts)
        )

    def measurementProcessingFinished(self):
        self.measurementProcessingThread.deleteLater()
        self.measurementProcessingThread = None
        self.measurementProcessingWorker = None
        self.processMeasurementsButton.setEnabled(True)
        self.combineMeasurementsButton.setEnabled(True)
        self.refreshButton.setEnabled(True)
        # Phase auto-apply (and clears) were postponed while processing was
        # running
        self.schedulePendingPhaseAutoApply()
        self.schedulePendingPhaseClears()
