"""Custom annotations of the measurements viewer
(ViewerCustomAnnotationsGuiElements), overriding the hooks of
gui/mixins/custom_annotations.py. Library:
measurements_viewer/custom_annotation_predictors.py. Spec:
measurements_viewer/__init__.py.
"""
from functools import partial
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QAction
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import qutils
from .. import custom_annotation_predictors
from .. import phase_translator
from cellacdc.gui.mixins.custom_annotations import custom_annot_symbol
from ..custom_annotation_predictors import CustomAnnotationPredictor
from cellacdc.timepoint_predictors import parse_flags
from .. import widgets as viewer_widgets


class ViewerCustomAnnotationsGuiElements:
    """The viewer custom annotations: overrides of the hooks of the shared
    CustomAnnotationsGuiElements (columns, shortcuts, tool tips, dialog
    kwargs), toggling cells, predictors, re-evaluation and bulk setting.
    The overrides that call super() (addCustomAnnotation,
    loadCustomAnnotations, reinitCustomAnnot,
    addCustomAnnnotScatterPlot) stay in
    measurements_viewer/main_window/window.py.
    """

    def updateReevaluateCustomAnnotationActions(self):
        hasPredictors = any(
            self.customAnnotHasPredictor(button)
            for button in self.customAnnotDict
        )
        self.reevaluateAllCustomAnnotationsAction.setEnabled(hasPredictors)
        self.reevaluateCellCustomAnnotationsAction.setEnabled(hasPredictors)

    def customAnnotIsVisibleMode(self):
        return bool(self.data)

    def defaultCustomAnnotShortcut(self, reservedShortcuts):
        for key in custom_annotation_predictors.DEFAULT_SHORTCUTS:
            if QKeySequence(key).toString() not in reservedShortcuts:
                return key
        return ''

    def customAnnotationDialogKwargs(self, state=None):
        posData = self.posData()
        button = (
            None if state is None 
            else self.customAnnotButtonByName(state['name'])
        )
        reservedShortcuts = self.reservedShortcuts(excludeCustomAnnot=button)
        kwargs = {
            'predictorOptions': {
                'acdc_df': self.dialogAnalysisDf(
                    self.timepointsDf(posData), posData
                ),
                'SizeT': self.dialogAnalysisSizeT(posData),
                'allChNames': getattr(posData, 'chNames', []),
                'isZstack': getattr(posData, 'SizeZ', 1) > 1,
                'isSegm3D': bool(self.isSegm3D),
                'validateDependencies': (
                    lambda name, predictor: self.validatePredictorDependencies(
                        None if state is None else state['name'],
                        name, predictor
                    )
                ),
            },
            # Checked on OK with the current keys (the dialog is not modal)
            'reservedShortcuts': partial(
                self.reservedShortcuts, excludeCustomAnnot=button
            ),
            'existingNames': [
                name for name in self.customAnnotNames()
                if state is None or name != state['name']
            ],
            # Checked on OK (the dialog stays open), the modified annotation
            # keeps its own column
            'columnConflict': (
                lambda name: ''
                if state is not None and name == state['name']
                else self.customAnnotColumnConflict(name)
            ),
        }
        if state is None:
            kwargs['state'] = {
                'type': 'Single time-point',
                'name': '',
                'symbol': '',
                'shortcut': self.defaultCustomAnnotShortcut(reservedShortcuts),
                'description': '',
            }
        return kwargs

    def customAnnotToolTip(self, state):
        shortcut = state.get('shortcut') or ''
        autoText = viewer_widgets.auto_shortcut_text(shortcut)
        if shortcut:
            autoDescr = (
                f'Auto: "{autoText}" (re-evaluate the predictor on the '
                'selected cell)' if autoText else 'Auto: none'
            )
            shortcutText = (
                f'Shortcut: "{shortcut}" (annotate/un-annotate the selected '
                f'cell across its entire video history)\n{autoDescr}'
            )
        else:
            shortcutText = 'No shortcut'
        predictor = CustomAnnotationPredictor.describe(state.get('predictor'))
        if state.get('autoApply', False):
            predictor = f'{predictor} (auto apply on unvisited cells)'
        lines = [
            f'Custom annotation: {state["name"]}', '',
            'Usage: select a cell and press the shortcut, or\n'
            'activate the button and RIGHT-click on a cell.',
            shortcutText, f'Predictor: {predictor}'
        ]
        if state.get('overridesPhases', False):
            lines.append(
                f'Overrides phases: annotated cells get the phase '
                f'"{state["name"]}" on all their frames (before the phase '
                'rules; the first overriding annotation of the toolbar set '
                'in a cell wins)'
            )
        description = state.get('description', '')
        if description:
            lines.extend(['', f'Description: {description}'])
        lines.extend([
            '', 'RIGHT-click on the button to modify,\n'
                're-evaluate, clear or remove the annotation.'
        ])
        # Names, descriptions and predictors are entered by the user
        return qutils.wrap_tooltip('\n'.join(lines))

    def setCustomAnnotShortcut(self, button, shortcut):
        """The shortcut annotates the selected cell across its entire history
        (instead of activating the button) and Shift+shortcut re-evaluates
        the predictor on the selected cell.

        The Shift+shortcut is not set if `shortcut` contains Shift or if it
        is already used (e.g., by a configuration created before it was
        reserved). Neither shortcut is set if `shortcut` itself is already
        used by another action (e.g., an annotation from 'Load
        annotation...' or a configuration, which skip the dialog checks),
        because Qt would trigger neither of two actions sharing a key. In
        both cases a warning is logged and a note is appended to the button
        tool tip, which must be set before calling this method."""
        button.setShortcut(QKeySequence())
        actions = self.customAnnotShortcutActions.get(button)
        if actions is None:
            toggleAction = QAction(self)
            toggleAction.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            toggleAction.triggered.connect(
                partial(self.customAnnotShortcutTriggered, button)
            )
            autoAction = QAction(self)
            autoAction.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            autoAction.triggered.connect(
                partial(self.reevaluateDisplayedCellCustomAnnot, button)
            )
            self.addAction(toggleAction)
            self.addAction(autoAction)
            actions = (toggleAction, autoAction)
            self.customAnnotShortcutActions[button] = actions
        toggleAction, autoAction = actions
        if isinstance(shortcut, QKeySequence):
            shortcut = shortcut.toString()
        keySequence = QKeySequence(shortcut or '')
        name = getattr(button, 'name', '')
        keyText = keySequence.toString()
        usedBy = None
        if keyText:
            usedBy = self.reservedShortcuts(
                excludeCustomAnnot=button, shiftDerived=False
            ).get(keyText)
        if usedBy is not None:
            self.logger.warning(
                f'Shortcut "{keyText}" of custom annotation '
                f'"{name}" is not set because it is already used by '
                f'"{usedBy}".'
            )
            button.setToolTip(qutils.wrap_tooltip(
                f'{button.toolTip()}\n\nNote: the shortcut "{keyText}" '
                f'(and its auto shortcut) is not set because it is already '
                f'used by "{usedBy}".'
            ))
            shortcut = ''
            keySequence = QKeySequence()
        autoText, autoUsedBy = self.autoShortcutConflict(
            shortcut, excludeCustomAnnot=button
        )
        if autoUsedBy is not None:
            self.logger.warning(
                f'Auto shortcut "{autoText}" of custom annotation '
                f'"{name}" is not set because it is already used by '
                f'"{autoUsedBy}".'
            )
            button.setToolTip(qutils.wrap_tooltip(
                f'{button.toolTip()}\n\nNote: the auto shortcut "{autoText}" '
                f'is not set because it is already used by "{autoUsedBy}".'
            ))
            autoText = ''
        toggleAction.setShortcut(keySequence)
        autoAction.setShortcut(QKeySequence(autoText))
        toggleAction.setText(f'Annotate "{name}"')
        autoAction.setText(f'Re-evaluate "{name}" predictor')

    def initCustomAnnotColumn(self, button, name):
        # Positions whose table already holds assignments of `name` (e.g.,
        # exported by an earlier session): their visited cells are not
        # auto-predicted (see `onCustomAnnotationChanged`)
        self.customAnnotPreexistingColumns[name] = [
            posData for posData in self.data or []
            if any(
                name in table.columns
                for table in self.positionTables(posData)
            )
        ]
        for acdc_df in self.allAcdcDfs():
            self.ensureCustomAnnotColumn(acdc_df, name)
        self.updateColumnsList()

    def renameCustomAnnotColumn(self, oldName, newName):
        self.renamePhaseTranslatorInput(oldName, newName)
        for acdc_df in self.allAcdcDfs():
            if oldName in acdc_df.columns:
                acdc_df.rename(columns={oldName: newName}, inplace=True)
        self.renameMeasurementColumnReferences(oldName, newName)
        self.plotMeasurements()

    def customAnnotColumnRemovalNote(self, name):
        # The column's own formatting is removed with it
        ownFormatting = f'conditional formatting of "{name}"'
        dependents = [
            dependent
            for dependent in self.columnDependents(
                name, excludeOutput=name, includePhaseInputs=True
            )
            if dependent != ownFormatting
        ]
        notes = []
        if dependents:
            notes.append(
                f'"{name}" is used by {", ".join(dependents)}, which will '
                'fail until edited if the column is removed.'
            )
        overrides = self.phaseOverrideAnnotations()
        if name in overrides:
            if not self.phaseTranslatorRules and overrides == [name]:
                # Nothing left to evaluate phases with
                # (`schedulePhaseOverrideCells`)
                notes.append(
                    f'"{name}" overrides phases: once it is removed, the '
                    f'phase "{name}" of its annotated cells is cleared.'
                )
            else:
                notes.append(
                    f'"{name}" overrides phases: once it is removed, its '
                    'annotated cells get their phases from the phase rules '
                    'again'
                    + (' (re-evaluated automatically).' if self.autoApplyPhases
                       else ' when phases are evaluated.')
                )
        if not notes:
            return ''
        return 'Note: ' + ' '.join(notes)

    def removeCustomAnnotColumn(self, name):
        for acdc_df in self.allAcdcDfs():
            acdc_df.drop(columns=name, errors='ignore', inplace=True)
        # The column's own plot settings go with it (`updateColumnsList`
        # drops it from the plotted columns and comparison axes)
        self.measurementsColors.pop(name, None)
        self._defaultSignalColorIndices.pop(name, None)
        self.conditionalFormatting.pop(name, None)
        self.timeCourseYAxes.pop(name, None)
        self.updateColumnsList()
        self.plotMeasurements()

    def onCustomAnnotationChanged(self, button, oldState=None):
        state = self.customAnnotDict[button]['state']
        name = state['name']
        oldName = name if oldState is None else oldState['name']
        if oldState is None:
            self.logger.info(f'Custom annotation "{name}" added')
        else:
            self.logger.info(
                f'Custom annotation "{oldName}" edited'
                + (f' (renamed to "{name}")' if oldName != name else '')
            )
        overrides = bool(state.get('overridesPhases', False))
        oldOverrides = oldState is not None and bool(
            oldState.get('overridesPhases', False)
        )
        if overrides != oldOverrides:
            self.logger.info(
                f'Custom annotation "{name}" '
                + ('overrides phases' if overrides else
                   'no longer overrides phases')
            )
        if oldName != name:
            for key in list(self.customAnnotAutoPredictions.keys()):
                if key[1] == oldName:
                    self.customAnnotAutoPredictions[(key[0], name)] = (
                        self.customAnnotAutoPredictions.pop(key)
                    )
            for posData in self.data or []:
                edited = self.editedCustomAnnotCells(posData)
                if oldName in edited:
                    edited[name] = edited.pop(oldName)
            if oldOverrides:
                # The color of the phase named after the annotation follows
                # the rename (unless a rule also assigns the old name)
                rulePhases = phase_translator.phase_names(
                    self.phaseTranslatorRules
                )
                if (oldName in self.phaseColors
                        and oldName not in rulePhases
                        and name not in self.phaseColors):
                    self.phaseColors[name] = self.phaseColors.pop(oldName)
        predictorChanged = (
            oldState is None 
            or oldState.get('predictor') != state.get('predictor')
        )
        preexisting = self.customAnnotPreexistingColumns.pop(name, [])
        if predictorChanged:
            self.clearCustomAnnotPredictionsTracking(name)
            # Like time points, automatic prediction only runs on cells not
            # viewed yet: a new or edited predictor keeps the (saved or
            # manual) assignments of visited cells. Only a brand-new column
            # is predicted on visited cells.
            if oldState is not None:
                self.seedVisitedCustomAnnotPredictions([name])
            elif preexisting:
                self.seedVisitedCustomAnnotPredictions([name], preexisting)
        elif (
            not oldState.get('autoApply', False)
            and state.get('autoApply', False)
        ):
            # Cells viewed while auto apply was off were never recorded as
            # predicted: enabling it keeps their assignments too.
            self.seedVisitedCustomAnnotPredictions([name])

        button.extraContextMenuActions = []
        if self.customAnnotHasPredictor(button):
            button.extraContextMenuActions.extend([
                (
                    'Re-evaluate predictor on all cells', 
                    self.reevaluateAllCellsCustomAnnot
                ),
                (
                    'Re-evaluate predictor on displayed cell', 
                    self.reevaluateDisplayedCellCustomAnnot
                ),
            ])
        button.extraContextMenuActions.append(
            ('Clear annotations of current position', self.clearCustomAnnotations)
        )
        self.updateReevaluateCustomAnnotationActions()
        if not self.data:
            return
        if (overrides or oldOverrides) and (
                overrides != oldOverrides or oldName != name):
            # The option was switched, an overriding annotation was renamed
            # (its cells get the new name) or a new overriding annotation
            # already has annotated cells: re-evaluate the annotated cells
            # and the cells with the annotation's (old) name as phase. Not
            # while annotations are set in bulk (configuration load).
            self.schedulePhaseOverrideCells([name], {oldName, name})
        self.ensureCustomAnnotPredictions()
        self.refreshCustomAnnotations([name])

    def onCustomAnnotationRemoved(self, button, state):
        for action in self.customAnnotShortcutActions.pop(button, ()):
            self.removeAction(action)
            # Still a child until deleted: free its key for the
            # `reservedShortcuts` check of annotations added right after
            action.setShortcut(QKeySequence())
            action.deleteLater()
        name = state['name']
        self.logger.info(f'Custom annotation "{name}" removed')
        self.clearCustomAnnotPredictionsTracking(name)
        for posData in self.data or []:
            self.editedCustomAnnotCells(posData).pop(name, None)
        self.updateReevaluateCustomAnnotationActions()
        if state.get('overridesPhases', False):
            # Its cells get their phases from the rules (or the next
            # overriding annotation) again: the cells with its name as
            # phase (its column may already be removed) and, if the column
            # is kept, the annotated cells. Not in bulk (`reinitCustomAnnot`)
            self.schedulePhaseOverrideCells([name], {name})

    def replaceCustomAnnotations(self, name, annotated, IDs, posData=None):
        """Replace the cell-wide assignments of IDs from predictor results.

        The phases of the cells whose assignment changed are re-evaluated
        (`schedulePhaseAutoApplyForAnnotation`)."""
        acdc_df = self.timepointsDf(posData)
        if acdc_df is None:
            return False
        if name in acdc_df.columns:
            values = parse_flags(acdc_df[name]).to_numpy().astype(int)
        else:
            values = np.zeros(len(acdc_df), dtype=int)
        previous = values.copy()
        cellIDs = acdc_df.index.get_level_values('Cell_ID')
        values[cellIDs.isin(list(IDs))] = 0
        markedIDs = {
            ID for ID, frames in annotated.items() if frames and ID in IDs
        }
        values[cellIDs.isin(markedIDs)] = 1
        acdc_df[name] = values
        changed = previous != values
        if changed.any():
            self.schedulePhaseAutoApplyForAnnotation(
                name, {int(ID) for ID in cellIDs[changed]}, posData
            )
        return True

    def refreshCustomAnnotations(self, names=None):
        """Redraw the annotations and the plot if it (or its conditional
        formatting) reads one of `names`."""
        self.doCustomAnnotation(0)
        if names is None:
            names = self.customAnnotNames()
        if self.plotReadsColumns(names):
            self.plotMeasurements()

    def toggleCustomAnnotation(self, button, ID):
        """Toggle one Cell_ID across its entire history in this position.

        Works on any frame, also where the cell is absent: the annotation is
        one value for the whole cell, so only a cell without any row in the
        measurements table is refused."""
        name = self.customAnnotDict[button]['state']['name']
        acdc_df = self.acdc_df
        if acdc_df is None:
            return False
        mask = acdc_df.index.get_level_values('Cell_ID') == ID
        if not mask.any():
            self.statusBar().showMessage(
                f'Cell ID {ID} is not in the measurements table.', 5000
            )
            return False
        self.ensureCustomAnnotColumn(acdc_df, name)
        annotated = bool(parse_flags(acdc_df.loc[mask, name]).any())
        acdc_df.loc[mask, name] = 0 if annotated else 1
        self.editedCustomAnnotCells()[name].add(int(ID))
        self.schedulePhaseAutoApplyForAnnotation(name, [int(ID)])
        action = 'Removed' if annotated else 'Added'
        self.logger.info(
            f'{action} custom annotation "{name}" for Cell ID {ID}'
        )
        self.statusBar().showMessage(
            f'{action} "{name}" annotation for Cell ID {ID} across its '
            'entire video history.', 5000
        )
        return True

    def customAnnotShortcutTriggered(self, button, checked=False):
        if not self.data or button not in self.customAnnotDict:
            return
        if self.selectedID is None:
            self.statusBar().showMessage('Select a cell to annotate.', 5000)
            return
        if self.toggleCustomAnnotation(button, int(self.selectedID)):
            self.refreshCustomAnnotations([button.name])

    def annotateClickedObject(self, xdata, ydata):
        """Annotate the object under the right-click with the active custom 
        annotation (as in `guiWin` right-click with an active button)."""
        button = self.customAnnotButton
        posData = self.posData()
        lab2D = self.get_2Dlab(posData.lab)
        try:
            ID = int(lab2D[ydata, xdata])
        except IndexError:
            ID = 0
        if ID == 0:
            self.statusBar().showMessage(
                'RIGHT-click on a cell to annotate it.', 5000
            )
            return
        self.doCustomAnnotation(ID)
        if not button.keepToolActive:
            button.setChecked(False)

    def doCustomAnnotation(self, ID):
        """Toggle the annotation of `ID` across its entire history with the
        active button (pass 0 to only redraw) and draw the annotations of 
        the current frame."""
        if not self.data or not self.customAnnotDict:
            return
        if self.switchPlaneCombobox.depthAxes() != 'z':
            self.clearCustomAnnot()
            return
        
        if ID > 0 and self.customAnnotButton in self.customAnnotDict:
            button = self.customAnnotButton
            if self.toggleCustomAnnotation(button, ID):
                if self.plotReadsColumns([button.name]):
                    self.plotMeasurements()
        
        if getattr(self.posData(), 'rp', None) is None:
            # Position still loading (e.g., predictions run by
            # `initIDsNavigation`): `updateImages` draws the annotations
            # once its objects are computed (`get_data`)
            return
        buttons = self.customAnnotButtonsToDraw()
        for button, annot in self.customAnnotDict.items():
            if button not in buttons:
                self.clearScatterPlotCustomAnnotButton(button)
                continue
            IDs = self.customAnnotFrameIDs(annot['state']['name'])
            self.drawCustomAnnotationIDs(button, IDs)
        if buttons:
            return buttons[0]

    def runCustomAnnotPredictor(
            self, button, posData=None, IDs=None, skipIDs=(), force=False
        ):
        """Same as `runTimepointPredictor` for a custom annotation: annotate 
        the entire cell history after boolean consolidation.
        
        If `IDs` is not None, only these cells are predicted. Cells in 
        `skipIDs` keep their current annotations. Unless `force` is True, 
        require `autoApply` and skip cells already predicted for the same 
        position path and annotation. Returns False if the prediction 
        failed."""
        if not self.data or button not in self.customAnnotDict:
            return False
        state = self.customAnnotDict[button]['state']
        if not force and not state.get('autoApply', False):
            return True
        if not self.customAnnotHasPredictor(button):
            return True
        if posData is None:
            posData = self.posData()
        acdc_df = self.timepointsDf(posData)
        if acdc_df is None:
            return True
        name = state['name']
        predictionKey = self.customAnnotPredictionKey(posData, name)
        alreadyPredicted = self.customAnnotAutoPredictions[predictionKey]
        allIDs = {
            int(ID) for ID in acdc_df.index.get_level_values('Cell_ID').unique()
        }
        IDs = allIDs if IDs is None else allIDs.intersection(IDs)
        IDs -= set(skipIDs)
        if not force:
            IDs -= alreadyPredicted
        if not IDs:
            return True
        try:
            # All cells are passed since predictors can use other cells
            # Only the analyzed frames (see `runTimepointPredictor`)
            predicted = CustomAnnotationPredictor.predict(
                self.analysisDf(acdc_df, posData), state['predictor'],
                SizeT=self.analysisSizeT(posData), IDs=IDs
            )
        except Exception as err:
            if not force:
                # Do not retry (and warn again) automatically
                alreadyPredicted.update(IDs)
            self.logger.exception('Custom annotation prediction failed')
            self.measurementProcessingWarning(
                'Custom annotation prediction failed', 
                f'Predicting "{name}" failed:\n\n{err}'
            )
            return False
        
        self.replaceCustomAnnotations(name, predicted, IDs, posData)
        self.editedCustomAnnotCells(posData)[name].difference_update(IDs)
        alreadyPredicted.update(IDs)
        
        numFrames = sum(len(frames) for frames in predicted.values())
        self.statusBar().showMessage(
            f'Predicted {numFrames} "{name}" annotation(s) for '
            f'{len(predicted)} cell(s).', 5000
        )
        return True

    def reevaluateCustomAnnotations(self, buttons=None, IDs=None, posData=None):
        """Re-run the predictors of the custom annotations `buttons` 
        (default all with a predictor) on cells `IDs` (default all cells), 
        warning if manual edits would be overwritten. Returns False if 
        cancelled."""
        if not self.data:
            return False
        if posData is None:
            posData = self.posData()
        if buttons is None:
            buttons = list(self.customAnnotDict.keys())
        buttons = [
            button for button in buttons
            if button in self.customAnnotDict 
            and self.customAnnotHasPredictor(button)
        ]
        if not buttons:
            self.statusBar().showMessage(
                'No custom annotation with a predictor.', 5000
            )
            return False
        columns = [self.customAnnotDict[button]['state']['name'] for button in buttons]
        self.logger.info(
            f'Re-evaluating custom annotations {", ".join(columns)} of '
            + ('all cells' if IDs is None else f'Cell ID(s) {list(IDs)}')
        )
        return self.evaluatePredictorColumns(columns, IDs, posData)

    def reevaluateAllCellsCustomAnnot(self, button=None, checked=False):
        buttons = None if button is None else [button]
        return self.reevaluateCustomAnnotations(buttons=buttons)

    def reevaluateAllCustomAnnotations(self, checked=False):
        return self.reevaluateAllCellsCustomAnnot()

    def reevaluateDisplayedCellCustomAnnotations(self, checked=False):
        return self.reevaluateDisplayedCellCustomAnnot()

    def reevaluateDisplayedCellCustomAnnot(self, button=None, checked=False):
        if not self.data:
            return False
        if self.selectedID is None:
            self.statusBar().showMessage('Select a cell first.', 5000)
            return False
        buttons = None if button is None else [button]
        return self.reevaluateCustomAnnotations(
            buttons=buttons, IDs=[int(self.selectedID)]
        )

    def clearCustomAnnotations(self, button, checked=False):
        """Remove all the annotations of `button` in the current position."""
        if not self.data or button not in self.customAnnotDict:
            return
        acdc_df = self.timepointsDf()
        if acdc_df is None:
            return
        name = self.customAnnotDict[button]['state']['name']
        self.logger.info(
            f'Clearing custom annotation "{name}" of all cells'
        )
        IDs = set(acdc_df.index.get_level_values('Cell_ID'))
        self.replaceCustomAnnotations(name, {}, IDs)
        self.editedCustomAnnotCells()[name] = set()
        self.refreshCustomAnnotations([name])

    def ensureCustomAnnotPredictions(self, posData=None):
        """Evaluate the automatic predictors of the custom annotations on 
        the displayed cell if not predicted yet."""
        self.ensurePredictorPredictions(posData)

    def setCustomAnnotations(self, states):
        """Replace the custom annotations with `states` (e.g., from a
        configuration). Requires loaded data.

        Adding them does not re-evaluate phases (the tables keep their
        saved phases, `bulkSettingCustomAnnotations`); the automatic
        predictions of the displayed cell, run afterwards, do when they
        change an annotation that affects phases."""
        self.setUncheckedAllCustomAnnotButtons()
        self.reinitCustomAnnot()
        if not self.data or not states:
            return
        self.readSavedCustomAnnot()
        with self.deferPredictorPredictions(), \
                self.bulkSettingCustomAnnotations():
            for savedState in states:
                symbolColor = pg.mkColor(tuple(savedState['symbolColor']))
                state = {
                    'type': 'Single time-point', 'symbol': '', 'shortcut': '',
                    'description': '', 'keepActive': True, 'isHideChecked': True,
                    **savedState, 'symbolColor': symbolColor
                }
                button = self.addCustomAnnotationItems(
                    custom_annot_symbol(state), symbolColor, state['shortcut'],
                    self.customAnnotToolTip(state), state['name'],
                    state['keepActive'], state['isHideChecked'], state
                )
                self.onCustomAnnotationChanged(button)
        self.saveCustomAnnot(only_temp=True)
