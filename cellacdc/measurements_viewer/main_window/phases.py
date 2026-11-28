"""Time point to phase translator UI and phase evaluation scheduling of the
measurements viewer (ViewerPhasesGuiElements). Library:
measurements_viewer/phase_translator.py. Spec:
measurements_viewer/__init__.py.
"""
import copy
from qtpy.QtCore import QTimer
from cellacdc import apps_measurements_viewer


class ViewerPhasesGuiElements:
    """The time point to phase translator UI, applying phases on the
    displayed cell or all cells, and scheduling automatic, override and
    pending phase evaluations and clears.
    """

    def setAutoApplyPhases(self, enabled):
        self.autoApplyPhases = bool(enabled)
        if self.autoApplyPhases:
            self.ensurePhasePrediction()

    def ensurePhasePrediction(self, posData=None, ID=None):
        """Schedule phase evaluation of the displayed cell (or `ID`) if it
        has no evaluated phase yet (no phase column or only empty phases).
        Each cell is tried once per session and position, so cells that no
        rule matches are not re-evaluated on every visit."""
        if not self.autoApplyPhases or not self.hasPhaseTranslation():
            return
        if not self.data:
            return
        if posData is None:
            posData = self.posData()
        if posData is not self.posData() or self.acdc_df is None:
            return
        if ID is None:
            ID = self.selectedID
        if ID is None:
            return
        ID = int(ID)
        tried = getattr(posData, 'phaseAutoEvaluatedIDs', None)
        if tried is None:
            tried = posData.phaseAutoEvaluatedIDs = set()
        if ID in tried:
            return
        tried.add(ID)
        try:
            df_ID = self.acdc_df.xs(ID, level='Cell_ID')
        except KeyError:
            return
        if self.phaseValues(df_ID).astype(bool).any():
            return
        self.schedulePhaseAutoApply([ID])

    def schedulePhaseAutoApply(self, IDs, posData=None):
        """Collect cells with changed time points (or custom annotations
        that affect phases, `annotationAffectsPhases`) and re-evaluate their
        phases once control returns to the event loop, so that batch
        updates (e.g., predictors on all cells) trigger one evaluation and
        callers holding a reference to the table are not disrupted.

        Pending cells are tracked per position (`posData`, default: the
        displayed one). Only the displayed position's cells are evaluated;
        those of another position (e.g., after switching position while
        measurement processing postponed the evaluation) stay pending
        until that position is displayed again, or are dropped when that
        position is no longer loaded (`prunePendingPhaseIDs`)."""
        if not self.autoApplyPhases or not self.hasPhaseTranslation():
            return
        if posData is None:
            if not self.data:
                return
            posData = self.posData()
        _, pending = self._pendingPhaseIDs.setdefault(
            id(posData), (posData, set())
        )
        pending.update(IDs)
        self.schedulePendingPhaseAutoApply()

    def schedulePendingPhaseAutoApply(self):
        """Start the (single) deferred phase auto-apply if the displayed
        position has pending cells."""
        if self._phaseAutoApplyScheduled or not self.data:
            return
        if id(self.posData()) not in self._pendingPhaseIDs:
            return
        self._phaseAutoApplyScheduled = True
        QTimer.singleShot(0, self.autoApplyPhaseTranslator)

    def autoApplyPhaseTranslator(self):
        self._phaseAutoApplyScheduled = False
        if self.measurementProcessingThread is not None:
            # Re-evaluated when measurement processing finishes
            return False
        if not self.autoApplyPhases or not self.hasPhaseTranslation():
            self._pendingPhaseIDs = {}
            return False
        self.prunePendingPhaseIDs()
        if not self.data or self.acdc_df is None:
            return False
        entry = self._pendingPhaseIDs.pop(id(self.posData()), None)
        if entry is None:
            return False
        IDs = entry[1].intersection(
            self.acdc_df.index.get_level_values('Cell_ID')
        )
        if not IDs:
            return False
        return self.applyPhaseTranslator(IDs=IDs, quiet=True)

    def configurePhaseTranslator(self, checked=False):
        posData = self.posData() if self.data else None
        # The rule formula tests see the frames used by the evaluation
        options = {
            'acdc_df': self.dialogAnalysisDf(self.acdc_df, posData),
            'SizeT': self.dialogAnalysisSizeT(posData),
            'allChNames': getattr(posData, 'chNames', []),
            'isZstack': getattr(posData, 'SizeZ', 1) > 1,
            'isSegm3D': bool(self.isSegm3D),
        }
        self.logger.info('Opening time point to phase translator...')
        dialog = apps_measurements_viewer.PhaseTranslatorDialog(
            self.phaseTranslatorRules,
            [self.timepointColumn(item) for item in self.timepointIdentifiers],
            self.customAnnotNames(), options, parent=self,
            phaseColors=self.phaseColors,
            annotationOverrides=self.phaseOverrideAnnotations()
        )

        dialog.autoApplyCheckbox.setChecked(self.autoApplyPhases)

        def apply(rules, allCells):
            # Apply commits the colors and options together with the rules.
            # Only the chosen colors: stored defaults would keep their
            # color while later defaults (by phase order) can repeat it
            self.phaseColors = dialog.chosenPhaseColors()
            self.setAutoApplyPhases(dialog.autoApplyCheckbox.isChecked())
            # The dialog is modal: its child message boxes stay clickable
            self.applyPhaseTranslator(rules, allCells, parent=dialog)

        dialog.sigApply.connect(apply)
        dialog.exec_()
        if not dialog.cancel:
            self.phaseTranslatorRules = copy.deepcopy(dialog.rules)
            self.phaseColors = dialog.chosenPhaseColors()
            self.setAutoApplyPhases(dialog.autoApplyCheckbox.isChecked())
            overrides = self.phaseOverrideAnnotations()
            self.logger.info(
                f'Phase translator rules set: {len(dialog.rules)} rule(s), '
                + (
                    'annotations overriding phases: '
                    + ', '.join(f'"{name}"' for name in overrides) + ', '
                    if overrides else ''
                )
                + f'auto apply {"on" if self.autoApplyPhases else "off"}'
            )
            self.plotMeasurements()
        else:
            self.logger.info('Phase translator closed without changes')
        dialog.deleteLater()

    def applyPhasesOnDisplayedCell(self, checked=False):
        """Evaluate the phases of the displayed cell with the configured
        phase translator rules and the custom annotations overriding phases
        (toolbar action and its shortcut)."""
        # Qt5 emits `triggered` also for a disabled action (shortcut)
        if not self.isDataLoaded or self.acdc_df is None:
            self.statusBar().showMessage('Load data first.', 5000)
            return False
        if not self.hasPhaseTranslation():
            self.statusBar().showMessage(
                'No phase rules or annotations overriding phases: add rules '
                'with "Time point to phase translator..." or enable '
                '"Overrides phases" of a custom annotation first.', 8000
            )
            self.logger.info(
                'Evaluate phases on displayed cell: no phase rules or '
                'annotations overriding phases defined'
            )
            return False
        return self.applyPhaseTranslator(allCells=False)

    def applyPhaseTranslator(
            self, rules=None, allCells=True, IDs=None, quiet=False,
            parent=None
        ):
        """Evaluate phases on all cells, the displayed cell (`allCells`
        False) or the cells `IDs` (overrides `allCells`) of the current
        position, with `rules` (default the configured ones) and the
        current custom annotations overriding phases
        (`phaseOverrideAnnotations`). With `quiet`, errors and warnings go
        to the status bar instead of message boxes (automatic
        re-evaluation). `parent` is the parent of the message boxes
        (default the viewer; the modal translator dialog when called from
        it)."""
        def warn(title, text):
            if quiet:
                self.statusBar().showMessage(f'{title}: {text}', 8000)
            else:
                self.measurementProcessingWarning(title, text, parent=parent)

        if self.measurementProcessingThread is not None:
            warn(
                'Measurement processing running',
                'Wait for measurement processing to finish before evaluating phases.'
            )
            return False
        if self.acdc_df is None or self.acdc_df.empty:
            warn('No measurements', 'Load measurements first.')
            return False
        if IDs is None and not allCells:
            if self.selectedID is None:
                warn('No cell selected', 'Select a cell first.')
                return False
            IDs = [self.selectedID]
        if rules is None:
            rules = self.phaseTranslatorRules
        overrides = self.phaseOverrideAnnotations()
        posData = self.posData()
        if not quiet:
            self.logger.info(
                'Evaluating phases of '
                + ('all cells' if IDs is None else f'Cell ID(s) {list(IDs)}')
                + f' ({len(rules)} rule(s)'
                + (
                    ', annotations overriding phases: '
                    + ', '.join(f'"{name}"' for name in overrides)
                    if overrides else ''
                )
                + ')...'
            )
        previousDtypes = list(self.acdc_df.dtypes.items())
        try:
            evaluation = self.evaluatePositionPhases(posData, IDs, rules)
        except Exception as err:
            self.logger.exception('Phase evaluation failed')
            warn('Phase evaluation failed', str(err))
            return False
        self.phaseTranslatorRules = copy.deepcopy(rules)
        # Rebuilding the columns tree is slow on large tables: only do it
        # when the evaluation added, removed or retyped columns
        columnsChanged = (
            list(evaluation.table.dtypes.items()) != previousDtypes
        )
        if columnsChanged:
            self.updateColumnsList()
        self.plotMeasurements()
        if evaluation.warnings:
            warn('Phase evaluation warnings', '\n'.join(evaluation.warnings))
            if quiet:
                return True
        if not quiet:
            self.logger.info('Phases evaluated')
        self.statusBar().showMessage(
            'Phases evaluated. Use Export measurements to save the table.', 5000
        )
        return True

    def schedulePhaseOverrideCells(self, names, phases=()):
        """Re-evaluate (deferred, `schedulePhaseAutoApply`) the cells of
        every loaded position annotated with `names` or with one of
        `phases`, e.g. after an annotation started or stopped overriding
        phases or was renamed or removed.

        When neither phase rules nor other overriding annotations are left
        (`hasPhaseTranslation`), nothing could evaluate these cells again:
        their `phases` are cleared instead (deferred,
        `clearPendingPhases`), also with automatic evaluation off."""
        if getattr(self, '_isBulkSettingCustomAnnotations', False):
            return
        if not self.hasPhaseTranslation():
            phases = set(phases)
            for posData in self.data or []:
                if not self.phaseOverrideCells(posData, (), phases):
                    continue
                _, pending = self._pendingPhaseClears.setdefault(
                    id(posData), (posData, set())
                )
                pending.update(phases)
            self.schedulePendingPhaseClears()
            return
        for posData in self.data or []:
            IDs = self.phaseOverrideCells(posData, names, phases)
            if IDs:
                self.schedulePhaseAutoApply(IDs, posData)

    def schedulePendingPhaseClears(self):
        """Start the (single) deferred `clearPendingPhases`."""
        if self._phaseClearScheduled or not self._pendingPhaseClears:
            return
        self._phaseClearScheduled = True
        QTimer.singleShot(0, self.clearPendingPhases)

    def clearPendingPhases(self, positions=None):
        """Clear the phases recorded by `schedulePhaseOverrideCells` (the
        names of removed or switched off overriding annotations) in the
        loaded `positions` (default: all), by evaluating their cells
        without rules and overrides (`phase_translator.evaluate` then
        resets them and drops the unused outputs, ``phase`` too once no
        cell keeps a phase). If
        phases can be evaluated again by now, the cells are scheduled for
        automatic evaluation instead. Postponed while measurement
        processing runs (`measurementProcessingFinished`)."""
        self._phaseClearScheduled = False
        if self.measurementProcessingThread is not None:
            return
        loaded = {id(posData): posData for posData in self.data or []}
        if positions is None:
            positions = list(loaded.values())
        displayedChanged = False
        for posData in positions:
            entry = self._pendingPhaseClears.pop(id(posData), None)
            if entry is None or loaded.get(id(posData)) is not entry[0]:
                continue
            phases = entry[1]
            IDs = self.phaseOverrideCells(posData, (), phases)
            if not IDs:
                continue
            if self.hasPhaseTranslation():
                self.schedulePhaseAutoApply(IDs, posData)
                continue
            try:
                self.evaluatePositionPhases(posData, IDs, rules=[])
            except Exception as err:
                self.logger.exception('Clearing phases failed')
                self.statusBar().showMessage(
                    f'Clearing phases failed: {err}', 8000
                )
                continue
            self.logger.info(
                'Cleared the phases '
                + ', '.join(f'"{phase}"' for phase in sorted(phases))
                + f' of Cell ID(s) {sorted(IDs)} of '
                + f'"{getattr(posData, "pos_foldername", "position")}"'
            )
            displayedChanged = displayedChanged or posData is self.posData()
        # Positions no longer loaded are forgotten
        self._pendingPhaseClears = {
            key: entry for key, entry in self._pendingPhaseClears.items()
            if loaded.get(key) is entry[0]
        }
        if displayedChanged:
            self.updateColumnsList()
            self.plotMeasurements()

    def flushPendingPhaseEvaluations(self, positions):
        """Run now the deferred phase evaluations of `positions` (automatic
        evaluation of changed cells, `schedulePhaseAutoApply`, which waits
        for a position to be displayed, and phase clears), e.g. before
        their tables are saved, so that no table is saved with phases that
        do not match its annotations. Skipped while measurement processing
        runs."""
        if self.measurementProcessingThread is not None or not self.data:
            return
        self.clearPendingPhases(positions)
        if not self.autoApplyPhases or not self.hasPhaseTranslation():
            return
        self.prunePendingPhaseIDs()
        displayed = self.posData()
        for posData in positions:
            if posData is displayed:
                # Updates the display too
                if id(displayed) in self._pendingPhaseIDs:
                    self.autoApplyPhaseTranslator()
                continue
            entry = self._pendingPhaseIDs.pop(id(posData), None)
            if entry is None:
                continue
            table = self.timepointsDf(posData)
            if table is None or table.empty:
                continue
            IDs = entry[1].intersection(
                table.index.get_level_values('Cell_ID')
            )
            if not IDs:
                continue
            try:
                self.evaluatePositionPhases(posData, IDs)
            except Exception as err:
                self.logger.exception('Phase evaluation failed')
                self.statusBar().showMessage(
                    f'Phase evaluation failed: {err}', 8000
                )

    def schedulePhaseAutoApplyForAnnotation(self, name, IDs, posData=None):
        """Re-evaluate (deferred) the phases of the cells `IDs` of `posData`
        (default: the displayed position) whose custom annotation `name`
        changed, if it affects phases (`annotationAffectsPhases`). Nothing is
        scheduled while annotations are set in bulk."""
        if getattr(self, '_isBulkSettingCustomAnnotations', False):
            return
        if not IDs or not self.annotationAffectsPhases(name):
            return
        self.schedulePhaseAutoApply(IDs, posData)
