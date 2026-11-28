"""Widget-free time point, custom annotation, predictor and phase
bookkeeping of the measurements viewer: the AnnotationDataUtils mixin (a
base of MeasurementsViewerUtils in measurements_viewer/viewer_utils.py),
the visited-cells column name and the 0/1 flag column checks of custom
annotations. Imports no module that imports it (it is a base-class
module of viewer_utils.py). Spec: measurements_viewer/__init__.py.
"""
import os
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import pandas as pd
from qtpy.QtGui import QColor
from . import phase_translator
from cellacdc import timepoint_predictors
from cellacdc.timepoint_predictors import TIMEPOINT_PREFIX, parse_flags


VISITED_COLUMN = 'measurements_viewer_visited' # for tracking which cells have been visited in the measurements viewer


def is_flag_column(values):
    """Whether the stored column `values` (Series) holds only 0/1 flags and
    missing values, so a custom annotation can reuse it: a bool-like column
    (`parse_flags` with ``strict=True``) or a numeric one whose values are
    all 0 or 1 (annotations are stored as int 0/1, which strict parsing
    rejects). Measurements and text columns are not flag columns."""
    if (pd.api.types.is_numeric_dtype(values.dtype)
            and not pd.api.types.is_bool_dtype(values.dtype)):
        return bool(values.dropna().isin([0, 1]).all())
    return parse_flags(values, strict=True) is not None


def flag_column_conflicts(names, tables):
    """Custom annotation `names` (in their order) whose column is in one of
    the `tables` (acdc_df, None skipped) but is not a 0/1 flag column
    (`is_flag_column`, e.g., a measurement): the annotation would turn its
    values into 0/1 (`ensureCustomAnnotColumn`)."""
    tables = [table for table in tables if table is not None]
    return [
        name for name in names
        if any(
            name in table.columns and not is_flag_column(table[name])
            for table in tables
        )
    ]


def flag_column_conflicts_text(names):
    """Why custom annotations `names` (`flag_column_conflicts`) cannot be
    used with the tables to load."""
    return (
        'The tables have columns named like the custom annotations '
        + ', '.join(f'"{name}"' for name in names)
        + '\nwith values other than 0/1 (e.g., a measurement), '
        'which the annotations would replace.'
    )


class AnnotationDataUtils:
    """Widget-free bookkeeping of time points, custom annotations,
    predictors and phases of the measurements viewer: time point columns
    and session state, predictor definitions and dependencies, custom
    annotation columns and conflicts (overrides of the shared GUI
    hooks), visited cells and auto-prediction tracking, phase
    evaluation, colors and runs. Must not call UI methods. Base of
    MeasurementsViewerUtils (measurements_viewer/viewer_utils.py).
    """

    def timepointColumn(self, identifier):
        return f'{TIMEPOINT_PREFIX}{identifier.name}'

    def ensureTimepointColumns(self, acdc_df):
        """Add the missing boolean column of each time point identifier."""
        for identifier in self.timepointIdentifiers:
            column = self.timepointColumn(identifier)
            if column not in acdc_df.columns:
                acdc_df[column] = False
            elif acdc_df[column].dtype != bool:
                acdc_df[column] = parse_flags(acdc_df[column])

    def timepointsDf(self, posData=None):
        """The acdc_df holding the time points of a position (or None)."""
        if posData is None:
            posData = self.posData()
        if posData is self.posData():
            return self.acdc_df
        return self.getAcdcDf(posData)

    def cellTimepoints(self, uid, ID, posData=None):
        """Sorted frame indices of the time points `uid` of cell `ID`."""
        acdc_df = self.timepointsDf(posData)
        column = self.timepointColumn(self.timepointIdentifier(uid))
        if acdc_df is None or column not in acdc_df.columns:
            return []
        try:
            values = acdc_df[column].xs(ID, level='Cell_ID')
        except KeyError:
            return []
        return sorted(int(frame_i) for frame_i in values.index[values.to_numpy()])

    def timepointCellIDs(self, uid, posData=None):
        """Cell IDs with at least one time point `uid`."""
        acdc_df = self.timepointsDf(posData)
        column = self.timepointColumn(self.timepointIdentifier(uid))
        if acdc_df is None or column not in acdc_df.columns:
            return set()
        IDs = acdc_df.index.get_level_values('Cell_ID')
        return set(IDs[acdc_df[column].to_numpy()])

    def prunePendingPhaseIDs(self):
        """Forget pending cells of positions that are no longer loaded (e.g.
        after loading other data), so their `posData` can be freed."""
        loaded = {id(posData): posData for posData in self.data or []}
        self._pendingPhaseIDs = {
            key: entry for key, entry in self._pendingPhaseIDs.items()
            if loaded.get(key) is entry[0]
        }

    def restoreTimepointColumns(self, posData, previous):
        """Keep the time points, the custom annotations, the visited cells
        and the phases evaluated in this session (which may differ from the
        file on disk) after the acdc_df of the position was reloaded.

        Phases (``posData.phaseTranslatorColumns``) are carried over only
        when they were evaluated in memory, so that they stay consistent
        with the kept time points: rows missing from `previous` get ``''``
        and translator outputs of the file dropped in memory are removed
        (all of them when every phase was cleared in memory: empty
        ``phaseTranslatorColumns``)."""
        acdc_df = getattr(posData, 'acdc_df', None)
        if previous is None or acdc_df is None or acdc_df.empty:
            return
        previous = self._standardizeAcdcDf(previous)
        acdc_df = self._standardizeAcdcDf(acdc_df)
        for identifier in self.timepointIdentifiers:
            column = self.timepointColumn(identifier)
            if column not in previous.columns:
                continue
            marked = previous.index[parse_flags(previous[column]).to_numpy()]
            acdc_df[column] = acdc_df.index.isin(marked)
        for name in self.customAnnotNames():
            if name not in previous.columns:
                continue
            marked = previous.index[parse_flags(previous[name]).to_numpy()]
            acdc_df[name] = acdc_df.index.isin(marked).astype(int)
            self.ensureCustomAnnotColumn(acdc_df, name)
        if VISITED_COLUMN in previous:
            marked = previous.index[parse_flags(previous[VISITED_COLUMN]).to_numpy()]
            acdc_df[VISITED_COLUMN] = acdc_df.index.isin(marked)
        owned = getattr(posData, 'phaseTranslatorColumns', None)
        if owned is not None and not owned:
            # Every phase was cleared in memory (evaluation without rules
            # and overrides removes all outputs)
            acdc_df = acdc_df.drop(columns=sorted(
                phase_translator.existing_output_columns(acdc_df)
            ))
        owned = set(owned or ())
        if 'phase' in owned and 'phase' in previous.columns:
            stale = {
                column for column in
                phase_translator.existing_output_columns(acdc_df)
                if column.startswith('phase_') and column not in owned
            }
            acdc_df = acdc_df.drop(columns=sorted(stale))
            phases = previous['phase'].reindex(acdc_df.index)
            acdc_df['phase'] = phases.where(phases.notna(), '').astype(object)
            for column in sorted(owned.difference({'phase'})):
                acdc_df[column] = acdc_df['phase'].eq(column[len('phase_'):])
        posData.acdc_df = acdc_df

    def evaluatePositionPhases(self, posData, IDs=None, rules=None):
        """Evaluate the phases of the cells `IDs` (None: all) of `posData`
        (any loaded position) with `rules` (default: the configured ones)
        and the custom annotations overriding phases, and store the result
        as the position's table (and working table, when displayed). The
        display is not updated. Raises on failure, without changes.

        Returns the `phase_translator.PhaseEvaluation`."""
        if rules is None:
            rules = self.phaseTranslatorRules
        table = self.timepointsDf(posData)
        if table is None or table.empty:
            raise ValueError('No measurements loaded for this position.')
        # Shallow copy: the ensure* methods only (re)assign whole columns,
        # which never writes into the table, and evaluate copies its input.
        # Avoids a full copy of the table on every time point edit.
        source = table.copy(deep=False)
        self.ensureTimepointColumns(source)
        self.ensureCustomAnnotColumns(source)
        evaluation = phase_translator.evaluate(
            source, rules, IDs=IDs,
            SizeT=getattr(posData, 'SizeT', None),
            owned_columns=getattr(posData, 'phaseTranslatorColumns', ()),
            last_frame=self.analysisLastFrame(posData),
            annotation_overrides=self.phaseOverrideAnnotations()
        )
        posData.phaseTranslatorColumns = evaluation.columns
        posData.acdc_df = evaluation.table
        if self.data and posData is self.posData():
            self.acdc_df = evaluation.table
        return evaluation

    def renamePhaseTranslatorInput(self, oldColumn, newColumn):
        for rule in self.phaseTranslatorRules:
            for key in ('timepoints', 'annotations'):
                rule[key] = [
                    newColumn if column == oldColumn else column
                    for column in rule[key]
                ]

    def timepointIdentifier(self, uid):
        for identifier in self.timepointIdentifiers:
            if identifier.uid == uid:
                return identifier

    def initPositionTimepointState(self, posData=None):
        """Session state of the time points of a position. The time points 
        themselves are stored in the acdc_df."""
        if posData is None:
            posData = self.posData()
        if getattr(posData, 'timepointsEdited', None) is None:
            # Cells with manual edits {uid: set of Cell_IDs}
            posData.timepointsEdited = defaultdict(set)

    def editedTimepointCells(self, posData=None):
        self.initPositionTimepointState(posData)
        if posData is None:
            posData = self.posData()
        return posData.timepointsEdited

    def markTimepointEdited(self, uid, ID):
        self.editedTimepointCells()[uid].add(ID)

    def predictorTimepointUids(self, uids=None):
        if uids is None:
            uids = [identifier.uid for identifier in self.timepointIdentifiers]
        return [
            uid for uid in uids 
            if self.timepointIdentifier(uid).predictor['type'] 
            != timepoint_predictors.PREDICTOR_NONE
        ]

    def predictorDefinitions(self):
        """Output column -> (kind, identifier/button, predictor, autoApply)."""
        definitions = {
            self.timepointColumn(identifier): (
                'timepoint', identifier.uid, identifier.predictor,
                identifier.autoApply
            )
            for identifier in self.timepointIdentifiers
        }
        for button, annot in self.customAnnotDict.items():
            state = annot['state']
            definitions[state['name']] = (
                'annotation', button,
                state.get('predictor') or {'type': timepoint_predictors.PREDICTOR_NONE},
                state.get('autoApply', False)
            )
        return definitions

    def columnDependents(
            self, column, excludeOutput=None, includePhaseInputs=False
        ):
        """Descriptions of the predictors, phase rule formulas, conditional
        formatting conditions and recorded combine/processing steps that
        read `column` (references are not rewritten on rename, so renaming
        is refused while any exist). The predictor of output
        `excludeOutput` is ignored. With `includePhaseInputs` (removal
        warnings), phase rules using `column` as a boundary or required
        annotation are listed too (renaming rewrites those)."""
        Dependencies = timepoint_predictors.PredictorDependencies
        dependents = []
        for output, definition in self.predictorDefinitions().items():
            if output == excludeOutput:
                continue
            if column in Dependencies.inputs(definition[2]):
                kind, key = definition[:2]
                dependents.append(
                    f'time point "{self.timepointIdentifier(key).name}"'
                    if kind == 'timepoint' else f'custom annotation "{output}"'
                )
        for rule in self.phaseRulesUsingColumn(column, includePhaseInputs):
            dependents.append(f'phase rule "{rule.get("name", "")}"')
        for formatted, rules in self.conditionalFormatting.items():
            if any(
                column in Dependencies.inputs({
                    'type': timepoint_predictors.PREDICTOR_EQUATION,
                    'equation': rule.get('condition', '')
                })
                for rule in rules
            ):
                dependents.append(f'conditional formatting of "{formatted}"')
        # Recorded steps are replayed on (re)load with the original column
        # names, so they also depend on the column keeping its name.
        for step in self.combinedMeasurementSteps:
            for name, expression in step.get('equations', {}).items():
                if column in Dependencies.inputs({
                    'type': timepoint_predictors.PREDICTOR_EQUATION,
                    'equation': expression
                }):
                    dependents.append(f'combined measurement "{name}"')
        for step in self.measurementProcessingSteps:
            if column in step.get('columns', []):
                outputs = ', '.join(step.get('output_columns', []))
                dependents.append(f'measurement processing step "{outputs}"')
        return dependents

    def phaseRulesUsingColumn(self, column, includePhaseInputs=False):
        """Phase rules whose eligibility formula reads `column` and, with
        `includePhaseInputs`, also those using it as a boundary or required
        annotation."""
        Dependencies = timepoint_predictors.PredictorDependencies
        rules = []
        for rule in self.phaseTranslatorRules:
            usesColumn = column in Dependencies.inputs(
                rule.get('predictor') or {}
            )
            if includePhaseInputs and not usesColumn:
                usesColumn = (
                    column in rule.get('timepoints', [])
                    or column in rule.get('annotations', [])
                )
            if usesColumn:
                rules.append(rule)
        return rules

    def validatePredictorDependencies(self, oldColumn, newColumn, predictor):
        definitions = self.predictorDefinitions()
        if newColumn != oldColumn and newColumn in definitions:
            raise ValueError(f'The predictor output "{newColumn}" already exists.')
        isRename = oldColumn in definitions and newColumn != oldColumn
        if isRename and any(
                newColumn in acdc_df.columns for acdc_df in self.allAcdcDfs()):
            # Renaming would duplicate the column. New identifiers and
            # annotations may reuse an existing (e.g., exported) column.
            raise ValueError(
                f'"{newColumn}" is already a column of the table. Choose a '
                'different name.'
            )
        if isRename:
            dependents = self.columnDependents(oldColumn, excludeOutput=oldColumn)
            if dependents:
                raise ValueError(
                    f'Cannot rename "{oldColumn}" because it is used by '
                    f'{", ".join(dependents)}. Edit them first (or keep the '
                    'name).'
                )
        predictors = {
            column: definition[2] for column, definition in definitions.items()
            if column != oldColumn
        }
        predictors[newColumn] = predictor
        timepoint_predictors.PredictorDependencies.order(predictors)

    def customAnnotNames(self):
        customAnnotDict = getattr(self, 'customAnnotDict', {})
        return [annot['state']['name'] for annot in customAnnotDict.values()]

    def phaseOverrideAnnotations(self):
        """Names of the custom annotations with the "Overrides phases"
        option (state key ``overridesPhases``), in toolbar (configuration)
        order: the first one set in a cell gives its phase
        (`phase_translator.evaluate` ``annotation_overrides``)."""
        customAnnotDict = getattr(self, 'customAnnotDict', {})
        return [
            annot['state']['name'] for annot in customAnnotDict.values()
            if annot['state'].get('overridesPhases', False)
        ]

    def hasPhaseTranslation(self):
        """Whether phases can be evaluated: at least one phase rule or one
        custom annotation overriding phases."""
        return bool(
            self.phaseTranslatorRules or self.phaseOverrideAnnotations()
        )

    def annotationAffectsPhases(self, name):
        """Whether changing the custom annotation `name` of a cell can change
        its phases: the annotation overrides phases, is a required
        annotation of a phase rule or is read by a rule's eligibility
        formula."""
        return (
            name in self.phaseOverrideAnnotations()
            or bool(self.phaseRulesUsingColumn(name, includePhaseInputs=True))
        )

    def phaseOverrideCells(self, posData, names, phases=()):
        """Cell_IDs of `posData` annotated with one of the annotations
        `names` (in any row, `parse_flags`) or with one of `phases` as phase
        in any row. Reads the tables without side effects
        (`positionTables`, unlike `timepointsDf`)."""
        IDs = set()
        for table in self.positionTables(posData):
            if table.empty:
                continue
            table = self._standardizeAcdcDf(table)
            cellIDs = table.index.get_level_values('Cell_ID')
            for name in names:
                if name in table.columns:
                    IDs.update(
                        int(ID) for ID in
                        cellIDs[parse_flags(table[name]).to_numpy()]
                    )
            if phases and 'phase' in table.columns:
                inPhases = np.isin(
                    self.phaseValues(table), list(phases)
                )
                IDs.update(int(ID) for ID in cellIDs[inPhases])
        return IDs

    @contextmanager
    def bulkSettingCustomAnnotations(self):
        """Annotations added or removed in bulk (configuration load,
        `reinitCustomAnnot`) do not re-evaluate phases: the loaded tables
        keep their saved phases."""
        previous = getattr(self, '_isBulkSettingCustomAnnotations', False)
        self._isBulkSettingCustomAnnotations = True
        try:
            yield
        finally:
            self._isBulkSettingCustomAnnotations = previous

    def ensureCustomAnnotColumn(self, acdc_df, name):
        """One 0/1 value per Cell_ID, broadcast to its entire history.

        Legacy frame-specific assignments mark the cell if any frame is
        annotated, preserving existing assignments.
        """
        if name not in acdc_df.columns:
            acdc_df[name] = 0
            return
        values = parse_flags(acdc_df[name])
        acdc_df[name] = (
            values.groupby(level='Cell_ID').transform('any').astype(int)
        )

    def ensureCustomAnnotColumns(self, acdc_df):
        for name in self.customAnnotNames():
            self.ensureCustomAnnotColumn(acdc_df, name)

    def customAnnotColumnConflict(self, name):
        """Why a new or loaded custom annotation cannot be called `name`
        ('' if it can): the column is written by a time point, another
        annotation, the phase translation, a recorded measurement step or
        the visited tracking, or it is a column of a loaded table that is
        not a 0/1 flag column (`is_flag_column`, e.g., a measurement), whose
        values `ensureCustomAnnotColumn` would turn into 0/1. Existing flag
        columns (e.g., exported annotations) are reused."""
        definitions = self.predictorDefinitions()
        if name in definitions:
            kind, key = definitions[name][:2]
            owner = (
                f'time point "{self.timepointIdentifier(key).name}"'
                if kind == 'timepoint' else f'custom annotation "{name}"'
            )
            return f'it is the column of the {owner}.'
        phaseColumns = set()
        if self.hasPhaseTranslation():
            phaseColumns = phase_translator.output_columns(
                self.phaseTranslatorRules, self.phaseOverrideAnnotations()
            )
        for posData in self.data or []:
            phaseColumns.update(
                getattr(posData, 'phaseTranslatorColumns', ()) or ()
            )
        if name in phaseColumns:
            return 'it is a column of the phases.'
        if name == VISITED_COLUMN:
            return 'it is the column of the visited cells.'
        steps = [
            *self.combinedMeasurementSteps, *self.measurementProcessingSteps
        ]
        if any(name in self.measurementStepOutputColumns(step) for step in steps):
            return 'it is the output of a recorded measurement step.'
        if flag_column_conflicts([name], self.allAcdcDfs()):
            return (
                'the table already has this column with values other '
                'than 0/1 (e.g., a measurement), which the annotation '
                'would replace.'
            )
        return ''

    def saveCustomAnnotPositionParams(self, posData):
        # Annotations are columns of the in-memory acdc_df (exported with 
        # the table) and their definitions are saved in configurations
        pass

    def saveCustomAnnot(self, only_temp=False):
        """Viewer definitions are saved by saveConfiguration, not globally."""
        pass

    def canDeleteSavedCustomAnnot(self):
        # saveCustomAnnot does not write the global file
        return False

    def visitedCellIDs(self, posData):
        """Cell_IDs of `posData` already viewed (`VISITED_COLUMN`)."""
        IDs = set()
        for table in self.positionTables(posData):
            if VISITED_COLUMN not in table:
                continue
            IDs.update(int(ID) for ID in table.index.get_level_values(
                'Cell_ID'
            )[parse_flags(table[VISITED_COLUMN]).to_numpy()])
        return IDs

    def resetAutoPredictionTracking(self):
        """Forget which cells were automatically predicted in this session
        (called when data is (re)loaded: the loaded tables' visited column
        is the only record of viewed cells, see the `seedVisited*`
        methods)."""
        self.timepointAutoPredictions.clear()
        self.customAnnotAutoPredictions.clear()

    def seedVisitedCustomAnnotPredictions(self, names, data=None):
        """Mark the visited cells of `data` (default all positions) as
        predicted for the annotations `names` so that automatic prediction
        keeps their assignments (explicit re-evaluation still applies)."""
        if data is None:
            data = self.data or []
        for posData in data:
            IDs = self.visitedCellIDs(posData)
            if not IDs:
                continue
            for name in names:
                self.customAnnotAutoPredictions[
                    self.customAnnotPredictionKey(posData, name)
                ].update(IDs)

    def seedVisitedTimepointPredictions(self, uids=None, data=None):
        """Mark the visited cells of `data` (default all positions) as
        predicted for the time point identifiers `uids` (default all) so
        that automatic prediction keeps their (saved or manual) time points,
        also when a dependent's other-cell inputs predict a prerequisite on
        all cells (explicit re-evaluation still applies)."""
        if uids is None:
            uids = [identifier.uid for identifier in self.timepointIdentifiers]
        if not uids:
            return
        if data is None:
            data = self.data or []
        for posData in data:
            IDs = self.visitedCellIDs(posData)
            if not IDs:
                continue
            path = os.path.normcase(os.path.abspath(posData.pos_path))
            for uid in uids:
                self.timepointAutoPredictions[(path, uid)].update(IDs)

    def clearCustomAnnotPredictionsTracking(self, name):
        for key in list(self.customAnnotAutoPredictions.keys()):
            if key[1] == name:
                self.customAnnotAutoPredictions.pop(key)

    def customAnnotHasPredictor(self, button):
        predictor = self.customAnnotDict[button]['state'].get('predictor')
        return (
            isinstance(predictor, dict) 
            and predictor.get('type', timepoint_predictors.PREDICTOR_NONE)
            != timepoint_predictors.PREDICTOR_NONE
        )

    def editedCustomAnnotCells(self, posData=None):
        """Cells with manual annotation edits {name: set of Cell_IDs}."""
        if posData is None:
            posData = self.posData()
        edited = getattr(posData, 'customAnnotEdited', None)
        if edited is None:
            edited = defaultdict(set)
            posData.customAnnotEdited = edited
        return edited

    def customAnnotPredictionKey(self, posData, name):
        return (os.path.normcase(os.path.abspath(posData.pos_path)), name)

    def customAnnotFrameIDs(self, name, frame_i=None, posData=None):
        """IDs annotated with `name` at `frame_i` (default current frame)."""
        acdc_df = self.timepointsDf(posData)
        if acdc_df is None or name not in acdc_df.columns:
            return []
        if frame_i is None:
            frame_i = self.frame_i
        try:
            values = acdc_df[name].xs(frame_i, level='frame_i')
        except KeyError:
            return []
        return [int(ID) for ID in values.index[parse_flags(values).to_numpy()]]

    def hasPhaseColumn(self):
        return self.acdc_df is not None and 'phase' in self.acdc_df.columns

    @staticmethod
    def phaseValues(df):
        """Phase name per row of df ('' if unassigned or no phase column)."""
        if 'phase' not in df.columns:
            return np.full(len(df), '', dtype=object)
        return df['phase'].map(
            lambda value: value if isinstance(value, str) else ''
        ).to_numpy(dtype=object)

    def phaseColorMap(self):
        """{phase name: QColor} of the rules' phases, then the names of the
        custom annotations overriding phases, and of any other phase found
        in the current table (e.g., from an exported CSV)."""
        names = phase_translator.phase_names(
            self.phaseTranslatorRules, self.phaseOverrideAnnotations()
        )
        if self.hasPhaseColumn():
            extra = set(self.phaseValues(self.acdc_df)).difference(names, {''})
            names += sorted(extra)
        return {
            name: QColor(color) for name, color in
            phase_translator.phase_colors(names, self.phaseColors).items()
        }

    @staticmethod
    def phaseRuns(frames, phases):
        """Contiguous runs [(start frame, end frame, phase)] of the same
        nonempty phase; a missing frame or a phase change ends a run."""
        runs = []
        for frame, phase in zip(frames, phases):
            if not phase:
                continue
            if runs and runs[-1][2] == phase and runs[-1][1] == frame - 1:
                runs[-1][1] = frame
            else:
                runs.append([frame, frame, phase])
        return [tuple(run) for run in runs]
