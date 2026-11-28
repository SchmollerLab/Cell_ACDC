"""Time point identifiers of the measurements viewer
(ViewerTimepointsGuiElements) and the predictor evaluation shared with
custom annotations (ViewerPredictorEvaluationGuiElements). Library:
timepoint_predictors.py. Spec: measurements_viewer/__init__.py.
"""
import os
from contextlib import contextmanager
from functools import partial
import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QIcon, QPixmap, QPainter, QCursor, QKeySequence
from qtpy.QtWidgets import QAction, QMenu, QAbstractButton
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import qutils
from cellacdc.shared.widgets import message_box as widgets_message_box
from cellacdc import html_utils
from cellacdc import timepoint_predictors
from ..dialogs import predictor_editors as viewer_dialogs_predictor_editors
from cellacdc.timepoint_predictors import (
    TimepointIdentifier, TIMEPOINT_PREFIX, parse_flags
)
from ..annotation_data import VISITED_COLUMN
from .. import widgets as viewer_widgets


class ViewerTimepointsGuiElements:
    """Time point identifiers: adding, editing and removing them, their
    toolbar actions and context menu, time point marks and lines on the
    plot, re-evaluation, and the reserved and auto (Shift+K) shortcuts
    shared with custom annotations.
    """

    def updateTimepointAndAnnotationShortcuts(self):
        for identifier in self.timepointIdentifiers:
            if identifier.uid in self.timepointActions:
                self.updateTimepointAction(identifier)
        for button, annotation in self.customAnnotDict.items():
            state = annotation['state']
            # `setCustomAnnotShortcut` appends its notes to the tool tip
            button.setToolTip(self.customAnnotToolTip(state))
            self.setCustomAnnotShortcut(button, state.get('shortcut'))

    def replaceTimepoints(self, uid, timepoints, IDs, posData=None):
        """Update one identifier's boolean column in the in-memory acdc_df.

        First clear all marks for cells in `IDs`, then mark every
        (frame_i, Cell_ID) pair supplied in `timepoints`. Other marks are
        preserved. Missing columns are initialized to False; pairs without
        an existing table row are ignored (no rows are added). This is the central
        function to update the time points inside acdc_df. 

        Parameters
        ----------
        uid : int
            UID of an existing time point identifier. Its name determines
            the column to update: ``timepoint_<name>``.
        timepoints : dict
            Mapping of Cell_ID to an iterable of zero-based frame indices.
            Normally its keys are a subset of `IDs`. Keys outside `IDs`
            also get marked, but their previous marks are not cleared.
        IDs : iterable of int
            Cells whose previous marks should be cleared. A cell omitted
            from `timepoints` has all its marks removed.
        posData : optional
            Target position; defaults to the currently displayed position.

        Returns
        -------
        bool
            False if the position has no table, otherwise True, even when
            none of the supplied pairs match a row.

        Notes
        -----
        This only changes the table in memory. It does not write a CSV,
        record manual edits, update prediction/viewed-cell tracking, or
        redraw plot lines; callers handle those steps separately.

        For example, ``replaceTimepoints(uid, {7: [2, 5]}, [7, 9])`` replaces
        cell 7's marks with frames 2 and 5 and clears all marks for cell 9,
        leaving other cells and other identifiers unchanged.
        """
        acdc_df = self.timepointsDf(posData)
        if acdc_df is None:
            return False
        column = self.timepointColumn(self.timepointIdentifier(uid))
        values = (
            acdc_df[column].to_numpy().copy() if column in acdc_df.columns 
            else np.zeros(len(acdc_df), dtype=bool)
        )
        values[acdc_df.index.get_level_values('Cell_ID').isin(list(IDs))] = False
        pairs = [
            (frame_i, ID) for ID, frames in timepoints.items() 
            for frame_i in frames
        ]
        if pairs:
            values |= acdc_df.index.isin(pairs)
        acdc_df[column] = values
        if posData is None or posData is self.posData():
            self.schedulePhaseAutoApply(set(IDs).union(timepoints))
        return True

    #         posData.cellConstants = {}
    #         posData.cellConstantsSource = None
    
    # def positionCellConstants(self, posData=None, acdc_df=None):
    #     """Values of the per-cell constants of a position 
    #     {name: {Cell_ID: value}} (or a Series indexed by (frame_i, Cell_ID) 
    #     for per-frame constants), recomputed if the constants or the 
    #     measurements table changed."""
    #     if posData is None:
    #         posData = self.posData()
    #     self.initPositionTimepointState(posData)
    #     if not self.cellConstants:
    #         posData.cellConstants = {}
    #         posData.cellConstantsSource = None
    #         return posData.cellConstants
    #     if acdc_df is None:
    #         acdc_df = (
    #             self.acdc_df if posData is self.posData() 
    #             else self.getAcdcDf(posData)
    #         )
    #     constants = tuple(
    #         tuple(timepoint_predictors.normalize_constant(constant).items())
    #         for constant in self.cellConstants
    #     )
    #     source = posData.cellConstantsSource
    #     if source is not None and source[0] is acdc_df and source[1] == constants:
    #         return posData.cellConstants
    #     try:
    #         values = timepoint_predictors.compute_cell_constants(
    #             acdc_df, self.cellConstants, SizeT=posData.SizeT
    #         )
    #     except Exception as err:
    #         self.logger.exception('Computing per-cell constants failed')
    #         self.measurementProcessingWarning(
    #             'Per-cell constants failed', 
    #             f'Computing the per-cell constants failed:\n\n{err}'
    #         )
    #         values = {}
    #     posData.cellConstants = values
    #     # Keep a reference to the table to detect changes by identity
    #     posData.cellConstantsSource = (acdc_df, constants)
    #     return values
    
    # def acdcDfWithConstants(self, acdc_df=None, posData=None):
    #     """Copy of `acdc_df` (default current table) with the per-cell 
    #     constants added as columns."""
    #     if not self.data:
    #         return acdc_df if acdc_df is not None else self.acdc_df
    #     if posData is None:
    #         posData = self.posData()
    #     if acdc_df is None and posData is self.posData():
    #         acdc_df = self.acdc_df
    #     if acdc_df is None:
    #         return None
    #     values = self.positionCellConstants(posData, acdc_df=acdc_df)
    #     return timepoint_predictors.add_constants_columns(acdc_df, values)
    
    # def setCellConstants(self, constants):
    #     timepoint_predictors.validate_constants(constants)
    #     self.cellConstants = [
    #         timepoint_predictors.normalize_constant(constant)
    #         for constant in constants
    #     ]
    #     for posData in self.data or []:
    #         if getattr(posData, 'timepointsPredicted', None) is not None:
    #             posData.cellConstants = {}
    #             posData.cellConstantsSource = None
    
    # def askReevaluateAfterConstantsChanged(self):
    #     txt = html_utils.paragraph(
    #         'The per-cell constants changed.<br><br>'
    #         'Do you want to <b>re-evaluate the time point predictors</b> on '
    #         'all cells now?'
    #     )
    #     msg = widgets.myMessageBox(wrapText=False)
    #     _, yesButton = msg.question(
    #         self, 'Re-evaluate predictors?', txt, 
    #         buttonsTexts=('No', 'Yes, re-evaluate')
    #     )
    #     return msg.clickedButton == yesButton
    
    # def editCellConstants(self, checked=False):
    #     posData = self.posData() if self.data else None
    #     dialog = apps.CellConstantsDialog(
    #         constants=self.cellConstants, acdc_df=self.acdc_df,
    #         SizeT=getattr(posData, 'SizeT', None), parent=self
    #     )
    #     dialog.exec_()
    #     if dialog.cancel or dialog.constants is None:
    #         return False
    #     newConstants = [
    #         timepoint_predictors.normalize_constant(constant) 
    #         for constant in dialog.constants
    #     ]
    #     if newConstants == self.cellConstants:
    #         return True
    #     self.setCellConstants(newConstants)
    #     if posData is not None:
    #         values = self.positionCellConstants(posData)
    #         self.statusBar().showMessage(
    #             f'Computed {len(values)} per-cell constant(s).', 5000
    #         )
    #     if posData is not None and self.predictorTimepointUids():
    #         # Other positions are re-predicted when they are loaded
    #         for otherPosData in self.data:
    #             if otherPosData is posData:
    #                 continue
    #             if getattr(otherPosData, 'timepointsPredicted', None) is not None:
    #                 otherPosData.timepointsPredicted.difference_update(
    #                     self.predictorTimepointUids()
    #                 )
    #                 positionPath = os.path.normcase(
    #                     os.path.abspath(otherPosData.pos_path)
    #                 )
    #                 for uid in self.predictorTimepointUids():
    #                     self.timepointAutoPredictions.pop((positionPath, uid), None)
    #         if self.askReevaluateAfterConstantsChanged():
    #             self.reevaluateAllTimepoints()
    #     return True
    
    def reservedShortcuts(
            self, excludeUid=None, excludeCustomAnnot=None, shiftDerived=True
        ):
        """Key sequences already used in the viewer {sequence: used by}.

        The shortcuts (including the Shift+ auto shortcut) of the time
        point identifier `excludeUid` and of the custom annotation button
        `excludeCustomAnnot` are not included (e.g., when editing them).

        If `shiftDerived` is True, a key K is also reserved when ``Shift+K``
        is used, because K would get ``Shift+K`` as auto shortcut (see
        `auto_shortcut_text`)."""
        reserved = {}
        for name in ('Left', 'Right', 'Up', 'Down', 'Space', 'Z', 'Esc'):
            reserved[name] = 'Viewer navigation'
        # `checkOverlayToolbuttonClicked` in `keyPressEvent` (0 = base layer)
        for n in range(10):
            reserved[str(n)] = 'Overlay channels toggle'
        zoomOutKey = QKeySequence(self.zoomOutKeyValue).toString()
        reserved[zoomOutKey] = 'Zoom out'
        timepointActions = {
            action: uid for uid, action in self.timepointActions.items()
        }
        timepointActions.update({
            action: uid for uid, action in self.timepointAutoActions.items()
        })
        excludedActions = set(
            self.customAnnotShortcutActions.get(excludeCustomAnnot, ())
        )
        widgets_ = self.findChildren(QAction) + self.findChildren(QAbstractButton)
        for widget in widgets_:
            if timepointActions.get(widget, -1) == excludeUid:
                continue
            if widget in excludedActions:
                continue
            shortcut = widget.shortcut().toString()
            if not shortcut:
                continue
            text = widget.text() or widget.toolTip().split('\n')[0]
            reserved[shortcut] = text.replace('&', '') or shortcut
        for name, widget in self.widgetsWithShortcut.items():
            shortcut = getattr(widget, 'keyPressShortcut', None)
            if isinstance(shortcut, QKeySequence) and not shortcut.isEmpty():
                reserved[shortcut.toString()] = name
        if not shiftDerived:
            return reserved
        for shortcut, usedBy in list(reserved.items()):
            if 'Shift+' not in shortcut:
                continue
            base = QKeySequence(shortcut.replace('Shift+', '', 1)).toString()
            if not base or base in reserved:
                continue
            if viewer_widgets.auto_shortcut_text(base) == shortcut:
                reserved[base] = f'{usedBy} ({shortcut})'
        return reserved

    def autoShortcutConflict(self, shortcut, excludeUid=None,
            excludeCustomAnnot=None):
        """Return (auto shortcut text, used by) of `shortcut`. `used by` is
        None if the auto shortcut is free (or empty)."""
        autoText = viewer_widgets.auto_shortcut_text(shortcut)
        if not autoText:
            return autoText, None
        reserved = self.reservedShortcuts(
            excludeUid=excludeUid, excludeCustomAnnot=excludeCustomAnnot
        )
        return autoText, reserved.get(autoText)

    def defaultTimepointIdentifier(self):
        names = {identifier.name for identifier in self.timepointIdentifiers}
        n = len(self.timepointIdentifiers) + 1
        while f'Time point {n}' in names:
            n += 1
        palette = timepoint_predictors.DEFAULT_COLORS
        usedColors = {
            tuple(identifier.color) for identifier in self.timepointIdentifiers
        }
        color = next(
            (c for c in palette if c not in usedColors), 
            palette[len(self.timepointIdentifiers) % len(palette)]
        )
        reserved = self.reservedShortcuts()
        shortcut = next(
            (key for key in timepoint_predictors.DEFAULT_SHORTCUTS 
             if key not in reserved), ''
        )
        return TimepointIdentifier(
            name=f'Time point {n}', color=color, shortcut=shortcut
        )

    def execTimepointIdentifierDialog(self, identifier, title):
        posData = self.posData() if self.data else None
        # Evaluated on OK: another identifier dialog may have added one
        # meanwhile (the dialog is not modal)
        names = lambda: {
            other.name for other in self.timepointIdentifiers
            if other.uid != identifier.uid
        }
        dialog = viewer_dialogs_predictor_editors.TimepointIdentifierDialog(
            identifier.name, identifier.color, identifier.shortcut,
            predictor=identifier.predictor, existingNames=names,
            reservedShortcuts=lambda: self.reservedShortcuts(
                excludeUid=identifier.uid
            ),
            acdc_df=(
                self.dialogAnalysisDf(self.timepointsDf(posData), posData)
                if posData is not None else None
            ),
            SizeT=self.dialogAnalysisSizeT(posData),
            allChNames=getattr(posData, 'chNames', []),
            isZstack=getattr(posData, 'SizeZ', 1) > 1,
            isSegm3D=bool(self.isSegm3D), title=title, parent=self
        )
        dialog.validatePredictorDependencies = (
            lambda name, predictor: self.validatePredictorDependencies(
                self.timepointColumn(identifier), f'{TIMEPOINT_PREFIX}{name}',
                predictor
            )
        )
        dialog.autoApplyCheckBox.setChecked(identifier.autoApply)
        dialog.exec_()
        # Deferred (result read below): a child of the viewer otherwise
        # keeps it and its nested editors (and their tables) alive
        dialog.deleteLater()
        if dialog.cancel:
            return
        return dialog.identifier

    def addTimepointIdentifier(self, checked=False):
        identifier = self.defaultTimepointIdentifier()
        values = self.execTimepointIdentifierDialog(
            identifier, 'Add time point identifier'
        )
        if values is None:
            return
        identifier = TimepointIdentifier(**values)
        column = self.timepointColumn(identifier)
        self.logger.info(
            f'Adding time point identifier "{identifier.name}" '
            f'(column "{column}")'
        )
        preexisting = [
            posData for posData in self.data or []
            if any(column in table for table in self.positionTables(posData))
        ]
        self.addTimepointIdentifierItem(identifier)
        # Saved time points of visited cells in an existing column are kept
        # (a brand-new column has none to keep).
        self.seedVisitedTimepointPredictions([identifier.uid], preexisting)
        self.ensureTimepointPredictions()
        self.updateColumnsList()
        self.plotMeasurements()
        self.updateTimepointLines()
        return identifier

    def addTimepointIdentifierItem(self, identifier):
        self.timepointIdentifiers.append(identifier)
        for acdc_df in self.allAcdcDfs():
            self.ensureTimepointColumns(acdc_df)
        # Shortcut: hovered plot frame, click: current frame
        action = viewer_widgets._ShortcutAwareAction(self)
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        action.triggered.connect(
            partial(self.timepointActionTriggered, identifier.uid)
        )
        self.timepointActions[identifier.uid] = action
        self.timepointsToolBar.addAction(action)
        
        # Shift+shortcut re-evaluates the predictor for the selected cell.
        autoAction = QAction(self)
        autoAction.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        autoAction.triggered.connect(
            lambda checked=False: self.reevaluateDisplayedCellTimepoints(
                uids=[identifier.uid]
            )
        )
        self.addAction(autoAction)
        self.timepointAutoActions[identifier.uid] = autoAction
        
        self.updateTimepointAction(identifier)
        self.updateReevaluateTimepointsActions()
        button = self.timepointsToolBar.widgetForAction(action)
        if button is not None:
            button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            button.customContextMenuRequested.connect(
                partial(self.showTimepointActionContextMenu, identifier.uid)
            )
        return action

    def updateTimepointAction(self, identifier):
        action = self.timepointActions[identifier.uid]
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(pg.mkPen(color=identifier.color, width=4))
        painter.drawLine(16, 2, 16, 30)
        painter.setBrush(pg.mkBrush(identifier.color))
        painter.drawEllipse(9, 9, 14, 14)
        painter.end()
        action.setIcon(QIcon(pixmap))
        action.setText(identifier.name)
        # A key already used by another action (e.g., a viewer tool key set
        # in the shortcuts editor or a configuration skipping the dialog
        # checks) is not set: Qt would trigger neither of the two actions
        keyText = QKeySequence(identifier.shortcut or '').toString()
        usedBy = None
        if keyText:
            usedBy = self.reservedShortcuts(
                excludeUid=identifier.uid, shiftDerived=False
            ).get(keyText)
        shortcut = identifier.shortcut if usedBy is None else ''
        if usedBy is not None:
            self.logger.warning(
                f'Shortcut "{keyText}" of time point "{identifier.name}" is '
                f'not set because it is already used by "{usedBy}".'
            )
        action.setShortcut(QKeySequence(shortcut or ''))
        autoText, autoUsedBy = self.autoShortcutConflict(
            shortcut, excludeUid=identifier.uid
        )
        if autoUsedBy is not None:
            self.logger.warning(
                f'Auto shortcut "{autoText}" of time point '
                f'"{identifier.name}" is not set because it is already used '
                f'by "{autoUsedBy}".'
            )
        if usedBy is not None:
            shortcutText = (
                f'No shortcut\nNote: the shortcut "{keyText}" (and its auto '
                f'shortcut) is not set because it is already used by '
                f'"{usedBy}".'
            )
        elif not identifier.shortcut:
            shortcutText = 'No shortcut'
        elif autoUsedBy is not None:
            shortcutText = (
                f'Shortcut: "{identifier.shortcut}", Auto: none '
                f'("{autoText}" is used by "{autoUsedBy}")'
            )
        elif autoText:
            shortcutText = (
                f'Shortcut: "{identifier.shortcut}", Auto: "{autoText}" '
                '(re-evaluate the predictor on the selected cell)'
            )
        else:
            shortcutText = f'Shortcut: "{identifier.shortcut}", Auto: none'
        predictorText = timepoint_predictors.TimepointPredictor.describe(
            identifier.predictor
        )
        action.setToolTip(qutils.wrap_tooltip(
            f'Time point: {identifier.name}\n\n'
            'Click to add it for the selected cell at the current frame.\n'
            'Hover over the measurements plot and press the shortcut to add '
            'it at the hovered frame.\n'
            'Right-click for more options.\n\n'
            f'{shortcutText}\nPredictor: {predictorText}'
        ))
        
        # update AutoAction
        autoAction = self.timepointAutoActions[identifier.uid]
        autoAction.setText(f'Re-evaluate "{identifier.name}" predictor')
        autoAction.setShortcut(
            QKeySequence(autoText if autoUsedBy is None else '')
        )


    def showTimepointActionContextMenu(self, uid, pos):
        identifier = self.timepointIdentifier(uid)
        button = self.timepointsToolBar.widgetForAction(
            self.timepointActions[uid]
        )
        menu = QMenu(button)
        menu.addAction('Edit...', partial(self.editTimepointIdentifier, uid))
        hasPredictor = (
            identifier.predictor['type'] != timepoint_predictors.PREDICTOR_NONE
        )
        predictAllAction = menu.addAction(
            'Re-evaluate predictor on all cells', 
            partial(self.reevaluateTimepoints, [uid])
        )
        predictCellAction = menu.addAction(
            'Re-evaluate predictor on displayed cell', 
            partial(self.reevaluateDisplayedCellTimepoints, uids=[uid])
        )
        predictAllAction.setEnabled(hasPredictor and bool(self.data))
        predictCellAction.setEnabled(
            hasPredictor and bool(self.data) and self.selectedID is not None
        )
        menu.addAction(
            'Clear time points of current position', 
            partial(self.clearTimepoints, uid)
        )
        menu.addSeparator()
        menu.addAction('Remove', partial(self.removeTimepointIdentifier, uid))
        menu.exec_(button.mapToGlobal(pos))

    def editTimepointIdentifier(self, uid):
        identifier = self.timepointIdentifier(uid)
        values = self.execTimepointIdentifierDialog(
            identifier, f'Edit time point identifier "{identifier.name}"'
        )
        if values is None:
            return
        if (self.timepointIdentifier(uid) is not identifier
                or uid not in self.timepointActions):
            # Removed (or replaced by a loaded configuration) while the
            # non-blocking dialog was open
            text = (
                f'The time point identifier "{identifier.name}" was removed '
                'while its dialog was open: changes not applied.'
            )
            self.logger.info(text)
            self.statusBar().showMessage(text, 8000)
            return
        predictorChanged = values['predictor'] != identifier.predictor
        oldColumn = self.timepointColumn(identifier)
        self.logger.info(
            f'Editing time point identifier "{identifier.name}"'
            + (f' (renamed to "{values["name"]}")'
               if values['name'] != identifier.name else '')
            + (' (predictor changed)' if predictorChanged else '')
        )
        identifier.name = values['name']
        newColumn = self.timepointColumn(identifier)
        if newColumn != oldColumn:
            self.renamePhaseTranslatorInput(oldColumn, newColumn)
            for acdc_df in self.allAcdcDfs():
                if oldColumn not in acdc_df.columns:
                    continue
                acdc_df.drop(columns=newColumn, inplace=True, errors='ignore')
                acdc_df.rename(columns={oldColumn: newColumn}, inplace=True)
            self.renameMeasurementColumnReferences(oldColumn, newColumn)
            self.plotMeasurements()
        identifier.color = values['color']
        identifier.shortcut = values['shortcut']
        identifier.predictor = values['predictor']
        identifier.autoApply = values['autoApply']
        self.updateTimepointAction(identifier)
        self.updateReevaluateTimepointsActions()
        if predictorChanged:
            for key in list(self.timepointAutoPredictions):
                if key[1] == uid:
                    del self.timepointAutoPredictions[key]
            # Visited cells keep their time points (as for annotations).
            self.seedVisitedTimepointPredictions([uid])
        self.updateTimepointLines()

    def removeTimepointIdentifier(self, uid, ask=True, updateColumns=True):
        identifier = self.timepointIdentifier(uid)
        if ask:
            column = self.timepointColumn(identifier)
            # The column's own formatting is removed with it
            ownFormatting = f'conditional formatting of "{column}"'
            dependents = [
                dependent
                for dependent in self.columnDependents(
                    column, excludeOutput=column, includePhaseInputs=True
                )
                if dependent != ownFormatting
            ]
            note = (
                f'\n\nNote: "{column}" is used by {", ".join(dependents)}, '
                'which will fail until edited.' if dependents else ''
            )
            msg = widgets_message_box.myMessageBox(wrapText=False)
            _, yes = msg.question(
                self, 'Remove time point identifier?',
                f'Remove "{identifier.name}" and its time points of all '
                f'positions?{note}', buttonsTexts=('Cancel', 'Remove')
            )
            if msg.clickedButton != yes:
                return
            self.logger.info(
                f'Removing time point identifier "{identifier.name}"'
            )
        action = self.timepointActions.pop(uid)
        self.timepointsToolBar.removeAction(action)
        # Still a child until deleted: free its key for `reservedShortcuts`
        action.setShortcut(QKeySequence())
        action.deleteLater()
        self.timepointIdentifiers.remove(identifier)
        for key in list(self.timepointAutoPredictions):
            if key[1] == uid:
                del self.timepointAutoPredictions[key]
        column = self.timepointColumn(identifier)
        for acdc_df in self.allAcdcDfs():
            acdc_df.drop(columns=column, inplace=True, errors='ignore')
        # The column's own plot settings go with it (as for custom
        # annotations). On load, the configuration's formatting is assigned
        # after `setTimepointIdentifiers`, so it is not wiped here.
        self.measurementsColors.pop(column, None)
        self._defaultSignalColorIndices.pop(column, None)
        self.conditionalFormatting.pop(column, None)
        self.timeCourseYAxes.pop(column, None)
        for posData in self.data:
            getattr(posData, 'timepointsEdited', {}).pop(uid, None)
            
        autoAction = self.timepointAutoActions.pop(uid)
        self.removeAction(autoAction)
        autoAction.setShortcut(QKeySequence())
        autoAction.deleteLater()

        self.updateReevaluateTimepointsActions()
        if updateColumns:
            self.updateColumnsList()
            self.plotMeasurements()
        self.updateTimepointLines()

    def clearTimepoints(self, uid):
        if not self.data:
            return
        self.logger.info(
            f'Clearing time points "{self.timepointIdentifier(uid).name}" '
            'of all cells'
        )
        self.replaceTimepoints(uid, {}, self.timepointCellIDs(uid))
        self.editedTimepointCells()[uid] = set()
        self.refreshPlotForColumns(
            [self.timepointColumn(self.timepointIdentifier(uid))]
        )

    def setTimepointIdentifiers(self, identifiers):
        for identifier in list(self.timepointIdentifiers):
            self.removeTimepointIdentifier(
                identifier.uid, ask=False, updateColumns=False
            )
        for identifier in identifiers:
            self.addTimepointIdentifierItem(identifier)
        self.updateColumnsList()
        self.plotMeasurements()

    def runTimepointPredictor(
            self, uid, posData=None, replot=False, IDs=None, skipIDs=(),
            force=False
        ):
        """Predict the time points of `uid` for the cells of `posData` 
        (default current position), replacing the existing ones.
        
        If `IDs` is not None, only these cells are predicted. Cells in 
        `skipIDs` keep their current time points. Unless `force` is True,
        require `autoApply` and skip cells already predicted in this session
        for the same position path and identifier. Returns False if the
        prediction failed."""
        identifier = self.timepointIdentifier(uid)
        if not self.data or identifier is None:
            return False
        if not force and not identifier.autoApply:
            return True
        
        if posData is None:
            posData = self.posData()
        self.initPositionTimepointState(posData)
        if identifier.predictor['type'] == timepoint_predictors.PREDICTOR_NONE:
            return True
        acdc_df = (
            self.acdc_df if posData is self.posData() 
            else self.getAcdcDf(posData)
        )
        predictionKey = (os.path.normcase(os.path.abspath(posData.pos_path)), uid)
        alreadyPredicted = self.timepointAutoPredictions[predictionKey]
        skipIDs = set(skipIDs)
        if IDs is not None:
            IDs = set(IDs) - skipIDs
        if not force:
            if IDs is None:
                IDs = (
                    set(acdc_df.index.get_level_values('Cell_ID'))
                    if acdc_df is not None else set()
                )
            IDs -= alreadyPredicted | skipIDs
        if IDs is not None and not IDs:
            return True
        try:
            # All cells are passed since predictors can use other cells
            # Only the analyzed frames (`analysisDf`): cut on every call,
            # since prerequisites write their outputs into the full table
            predicted = timepoint_predictors.TimepointPredictor.predict(
                self.analysisDf(acdc_df, posData), identifier.predictor,
                SizeT=self.analysisSizeT(posData), IDs=IDs
            )
        except Exception as err:
            self.logger.exception('Time point prediction failed')
            self.measurementProcessingWarning(
                'Time point prediction failed', 
                f'Predicting "{identifier.name}" failed:\n\n{err}'
            )
            return False
        
        if IDs is None:
            replacedIDs = (
                self.timepointCellIDs(uid, posData).union(predicted) - skipIDs
            )
        else:
            replacedIDs = set(IDs) - skipIDs
        self.replaceTimepoints(
            uid, {ID: predicted[ID] for ID in replacedIDs if ID in predicted},
            replacedIDs, posData
        )
        posData.timepointsEdited[uid].difference_update(replacedIDs)

        completedIDs = (
            set(acdc_df.index.get_level_values('Cell_ID'))
            if acdc_df is not None else set()
        )
        if IDs is not None:
            completedIDs.intersection_update(IDs)
        alreadyPredicted.update(completedIDs - skipIDs)

        numCells = sum(1 for ID in replacedIDs if ID in predicted)
        numTimepoints = sum(len(predicted[ID]) for ID in replacedIDs if ID in predicted)
        self.statusBar().showMessage(
            f'Predicted {numTimepoints} "{identifier.name}" time point(s) '
            f'for {numCells} cell(s).', 5000
        )
        if replot:
            self.refreshPlotForColumns([self.timepointColumn(identifier)])
        return True

    def updateReevaluateTimepointsActions(self):
        hasPredictors = bool(self.predictorTimepointUids())
        self.reevaluateAllTimepointsAction.setEnabled(hasPredictors)
        self.reevaluateCellTimepointsAction.setEnabled(hasPredictors)

    def reevaluateTimepoints(self, uids=None, IDs=None, posData=None):
        """Re-run the predictors of `uids` (default all identifiers with 
        a predictor) on cells `IDs` (default all cells) of `posData`, 
        warning if manual edits would be overwritten. Returns False if 
        cancelled."""
        if not self.data:
            return False
        if posData is None:
            posData = self.posData()
        uids = self.predictorTimepointUids(uids)
        if not uids:
            self.statusBar().showMessage(
                'No time point identifier with a predictor.', 5000
            )
            return False
        columns = [
            self.timepointColumn(self.timepointIdentifier(uid)) for uid in uids
        ]
        self.logger.info(
            f'Re-evaluating time points {", ".join(columns)} of '
            + ('all cells' if IDs is None else f'Cell ID(s) {list(IDs)}')
        )
        return self.evaluatePredictorColumns(columns, IDs, posData)

    def reevaluateAllTimepoints(self, checked=False):
        return self.reevaluateTimepoints()

    def reevaluateDisplayedCellTimepoints(self, checked=False, uids=None):
        if self.selectedID is None:
            self.statusBar().showMessage('Select a cell first.', 5000)
            return False
        return self.reevaluateTimepoints(uids=uids, IDs=[self.selectedID])

    def ensureTimepointPredictions(self, posData=None):
        """Evaluate automatic predictors only on the newly displayed cell."""
        self.ensurePredictorPredictions(posData)

    def hoveredMeasurementsPlotFrame(self):
        """Frame index under the mouse cursor in the measurements plot or 
        None if the cursor is not over the plot."""
        plot = self.measurementsPlot
        if not self.data or not plot.isVisible():
            return
        pos = plot.mapFromGlobal(QCursor.pos())
        if not plot.rect().contains(pos):
            return
        viewPos = plot.getViewBox().mapSceneToView(plot.mapToScene(pos))
        return self.clipFrame(int(round(viewPos.x())) - 1)

    def timepointActionTriggered(self, uid, checked=False):
        """Add a time point of the identifier `uid` for the selected cell.

        Its shortcut adds it at the frame under the mouse cursor in the
        measurements plot (nothing when the plot is not hovered). A click on
        its toolbar button (or any other trigger) adds it at the current
        frame. The shortcut is recognised by the action
        (`_ShortcutAwareAction`), not by the cursor position, which may be
        anywhere when the button is clicked (e.g., toolbar extension).
        """
        if not self.data:
            self.statusBar().showMessage('Load data to add time points.', 5000)
            return
        identifier = self.timepointIdentifier(uid)
        action = self.timepointActions.get(uid)
        if getattr(action, 'triggeredByShortcut', False):
            frame_i = self.hoveredMeasurementsPlotFrame()
            if frame_i is None:
                self.statusBar().showMessage(
                    f'Hover over the measurements plot to add a '
                    f'"{identifier.name}" time point with its shortcut, or '
                    'click its button to add it at the current frame.', 5000
                )
                return
        else:
            frame_i = self.frame_i
        self.addTimepoint(uid, frame_i)

    def addTimepoint(self, uid, frame_i, ID=None):
        identifier = self.timepointIdentifier(uid)
        if not self.data or frame_i is None:
            self.statusBar().showMessage('Load data to add time points.', 5000)
            return False
        if self.measurementsModeCombobox.currentIndex() == 1:
            self.statusBar().showMessage(
                'Switch to "Time course" mode to add time points.', 5000
            )
            return False
        if ID is None:
            ID = self.selectedID
        if ID is None:
            self.statusBar().showMessage('Select a cell to add time points.', 5000)
            return False
        frame_i = self.clipFrame(frame_i)
        cellTimepoints = self.cellTimepoints(uid, ID)
        if frame_i in cellTimepoints:
            self.statusBar().showMessage(
                f'"{identifier.name}" is already at frame n. {frame_i+1} '
                f'for Cell ID {ID}.', 5000
            )
            return False
        if not self.cellHasRow(ID, frame_i):
            self.statusBar().showMessage(
                f'Cell ID {ID} is not in the measurements table at frame n. '
                f'{frame_i+1}.', 5000
            )
            return False
        self.logger.info(
            f'Adding time point "{identifier.name}" at frame n. {frame_i+1} '
            f'for Cell ID {ID}'
        )
        self.replaceTimepoints(uid, {ID: cellTimepoints + [frame_i]}, [ID])
        self.markTimepointEdited(uid, ID)
        self.refreshPlotForColumns([self.timepointColumn(identifier)])
        self.statusBar().showMessage(
            f'Added "{identifier.name}" at frame n. {frame_i+1} for '
            f'Cell ID {ID}.', 5000
        )
        return True

    def removeTimepoint(self, uid, ID, frame_i):
        frames = [f for f in self.cellTimepoints(uid, ID) if f != frame_i]
        self.replaceTimepoints(uid, {ID: frames}, [ID])

    def timepointLineRemoveRequested(self, line):
        self.logger.info(
            f'Removing time point "{self.timepointIdentifier(line.uid).name}" '
            f'at frame n. {line.frame_i+1} of Cell ID {line.ID}'
        )
        self.removeTimepoint(line.uid, line.ID, line.frame_i)
        self.markTimepointEdited(line.uid, line.ID)
        self.refreshPlotForColumns(
            [self.timepointColumn(self.timepointIdentifier(line.uid))]
        )

    def timepointLineMoved(self, line):
        new_frame_i = self.clipFrame(int(round(line.value())) - 1)
        editedColumns = []
        # A time point can only sit on a frame where the cell has a row
        if (new_frame_i != line.frame_i
                and self.cellHasRow(line.ID, new_frame_i)):
            self.logger.info(
                'Moving time point '
                f'"{self.timepointIdentifier(line.uid).name}" of Cell ID '
                f'{line.ID} from frame n. {line.frame_i+1} to '
                f'{new_frame_i+1}'
            )
            self.removeTimepoint(line.uid, line.ID, line.frame_i)
            frames = self.cellTimepoints(line.uid, line.ID)
            self.replaceTimepoints(
                line.uid, {line.ID: frames + [new_frame_i]}, [line.ID]
            )
            self.markTimepointEdited(line.uid, line.ID)
            editedColumns.append(
                self.timepointColumn(self.timepointIdentifier(line.uid))
            )
        # Rebuild the lines after the drag event has been fully handled
        QTimer.singleShot(
            0, partial(self.refreshPlotForColumns, editedColumns)
        )

    def updateTimepointLines(self):
        for line in self.timepointLines:
            self.measurementsPlot.removeItem(line)
        self.timepointLines = []
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        if not self.data or isComparison or self.selectedID is None:
            return
        for identifier in self.timepointIdentifiers:
            frames = self.cellTimepoints(identifier.uid, self.selectedID)
            for frame_i in frames:
                line = viewer_widgets._TimepointLine(identifier, self.selectedID, frame_i)
                line.sigPositionChangeFinished.connect(self.timepointLineMoved)
                line.sigRemoveRequested.connect(
                    self.timepointLineRemoveRequested
                )
                self.measurementsPlot.addItem(line, ignoreBounds=True)
                self.timepointLines.append(line)


class ViewerPredictorEvaluationGuiElements:
    """Evaluation of time point and custom annotation predictors shared by
    both features: deferred predictions, the overwrite prompt for manual
    edits and evaluating the predictor columns.
    """

    def askOverwriteManualTimepoints(self, editedCells, scope, what='Time points'):
        """Warn that re-evaluating overwrites manual edits.
        
        Parameters
        ----------
        editedCells : dict
            {identifier name: sorted list of edited Cell_IDs}
        scope : str
            Description of the re-evaluated cells.
        what : str, optional
            What has been edited (e.g., 'Custom annotations').

        Returns
        -------
        str
            'overwrite', 'skip' (keep the manually edited cells) or 'cancel'.
        """
        details = '\n'.join(
            f'{name}: Cell ID {", ".join(str(ID) for ID in IDs)}'
            for name, IDs in editedCells.items()
        )
        txt = html_utils.paragraph(
            f'{what} of {scope} have been <b>edited manually</b>.<br><br>'
            'Re-evaluating the predictors will <b>overwrite</b> these edits.'
            '<br><br>Do you want to overwrite the manual edits or skip the '
            'edited cells?'
        )
        msg = widgets_message_box.myMessageBox(wrapText=False)
        _, skipButton, overwriteButton = msg.warning(
            self, 'Overwrite manual edits?', txt, detailsText=details,
            buttonsTexts=('Cancel', 'Skip edited cells', 'Overwrite')
        )
        if msg.clickedButton == overwriteButton:
            return 'overwrite'
        if msg.clickedButton == skipButton:
            return 'skip'
        return 'cancel'

    @contextmanager
    def deferPredictorPredictions(self):
        previous = getattr(self, '_deferPredictorPredictions', False)
        self._deferPredictorPredictions = True
        try:
            yield
        finally:
            self._deferPredictorPredictions = previous
        if not previous:
            self.ensurePredictorPredictions()

    def evaluatePredictorColumns(
            self, columns, IDs, posData, automatic=False
        ):
        definitions = self.predictorDefinitions()
        predictors = {
            column: definition[2] for column, definition in definitions.items()
            if definition[2].get('type') != timepoint_predictors.PREDICTOR_NONE
        }
        try:
            ordered = timepoint_predictors.PredictorDependencies.order(
                predictors, columns
            )
        except ValueError as err:
            self.measurementProcessingWarning('Invalid predictors', str(err))
            return False
        if automatic:
            ordered = [column for column in ordered if definitions[column][3]]
        scopes = {column: None if IDs is None else set(IDs) for column in ordered}
        for column in reversed(ordered):
            inputs = timepoint_predictors.PredictorDependencies.inputs(
                predictors[column]
            )
            for dependency, otherCells in inputs.items():
                if dependency in scopes and (
                        otherCells or scopes[column] is None):
                    scopes[dependency] = None
        edited = {}
        editedCells = {}
        for column in ordered:
            kind, key, _, _ = definitions[column]
            cells = (
                self.editedTimepointCells(posData)[key]
                if kind == 'timepoint' else
                self.editedCustomAnnotCells(posData)[column]
            )
            scopeIDs = scopes[column]
            edited[column] = set(cells) if scopeIDs is None else cells & scopeIDs
            if edited[column]:
                name = (
                    self.timepointIdentifier(key).name
                    if kind == 'timepoint' else column
                )
                editedCells[name] = sorted(edited[column])
        # Automatic predictions never overwrite manual edits (no prompt):
        # edited cells, including those of prerequisites, are skipped
        skip = automatic
        if editedCells and not automatic:
            allEdited = set().union(*editedCells.values())
            scope = (
                f'the displayed cell (ID {next(iter(IDs))})'
                if IDs is not None and len(IDs) == 1 and allEdited <= set(IDs)
                else f'{len(allEdited)} cell(s)'
            )
            if posData is not self.posData():
                posName = getattr(posData, 'pos_foldername', 'another position')
                scope = f'{scope} of position "{posName}"'
            kinds = {definitions[column][0] for column in ordered}
            if kinds == {'timepoint'}:
                answer = self.askOverwriteManualTimepoints(editedCells, scope)
            else:
                what = (
                    'Custom annotations' if kinds == {'annotation'}
                    else 'Time points and custom annotations'
                )
                answer = self.askOverwriteManualTimepoints(
                    editedCells, scope, what=what
                )
            if answer == 'cancel':
                return False
            skip = answer == 'skip'
        failed = set()
        names = []
        for column in ordered:
            inputs = timepoint_predictors.PredictorDependencies.inputs(
                predictors[column]
            )
            if failed.intersection(inputs):
                failed.add(column)
                self.statusBar().showMessage(
                    f'Skipped "{column}": a prerequisite prediction failed.', 5000
                )
                continue
            kind, key, _, _ = definitions[column]
            run = (
                self.runTimepointPredictor if kind == 'timepoint'
                else self.runCustomAnnotPredictor
            )
            if not run(
                    key, posData=posData, IDs=scopes[column],
                    skipIDs=edited[column] if skip else (),
                    force=not automatic
                ):
                failed.add(column)
            if kind == 'annotation':
                names.append(column)
        if automatic and failed:
            path = os.path.normcase(os.path.abspath(posData.pos_path))
            acdc_df = self.timepointsDf(posData)
            for column in failed:
                scopeIDs = scopes[column]
                if scopeIDs is None:
                    scopeIDs = (
                        set(acdc_df.index.get_level_values('Cell_ID'))
                        if acdc_df is not None else set()
                    )
                kind, key, _, _ = definitions[column]
                predictions = (
                    self.timepointAutoPredictions if kind == 'timepoint'
                    else self.customAnnotAutoPredictions
                )
                predictions[(path, key if kind == 'timepoint' else column)].update(
                    scopeIDs
                )
        if names:
            # Redraw the annotations, the plot is refreshed below
            self.doCustomAnnotation(0)
        self.refreshPlotForColumns(ordered)
        return not failed

    def ensurePredictorPredictions(self, posData=None):
        if getattr(self, '_deferPredictorPredictions', False):
            return
        if not self.data or self.selectedID is None:
            return
        if posData is None:
            posData = self.posData()
        if posData is not self.posData():
            return
        ID = int(self.selectedID)
        acdc_df = self.timepointsDf(posData)
        if acdc_df is None:
            return
        if VISITED_COLUMN not in acdc_df:
            acdc_df[VISITED_COLUMN] = False
        rows = acdc_df.index.get_level_values('Cell_ID') == ID
        newCell = not parse_flags(acdc_df.loc[rows, VISITED_COLUMN]).any()
        acdc_df.loc[rows, VISITED_COLUMN] = True
        posData.acdc_df = acdc_df
        self.initPositionTimepointState(posData)
        columns = []
        for column, definition in self.predictorDefinitions().items():
            kind, key, predictor, autoApply = definition
            if (not autoApply
                    or predictor.get('type') == timepoint_predictors.PREDICTOR_NONE):
                continue
            if kind == 'timepoint':
                if not newCell:
                    continue
            elif ID in self.customAnnotAutoPredictions[
                    self.customAnnotPredictionKey(posData, column)]:
                continue
            columns.append(column)
        if columns:
            self.evaluatePredictorColumns(columns, [ID], posData, automatic=True)
        self.ensurePhasePrediction(posData, ID)
