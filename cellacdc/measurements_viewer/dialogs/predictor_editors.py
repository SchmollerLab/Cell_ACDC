"""Editors of measurements viewer predictors: time point identifiers
(TimepointIdentifierDialog), time point and custom annotation predictor
equations, the main/other-cell filter equations, the custom annotation
predictor group box, and disabled_while_nested for nested editors. The
custom annotation dialog in gui/mixins/dialogs.py imports the group box
lazily, only when the viewer passes predictor options. Equation syntax:
timepoint_predictors.py and docs/source/equation-editor.rst.
"""
import configparser
import copy
import html
import numpy as np
import os
import pandas as pd
import pyqtgraph as pg
import re
import traceback
from contextlib import contextmanager
from functools import partial
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from cellacdc import apps, background_values, fonts, html_utils, myutils
from .. import phase_translator
from cellacdc import (
    qutils, settings_folderpath, timepoint_predictors, urls, widgets
)
from cellacdc.timepoint_predictors import TIMEPOINT_PREFIX, debug_trace as timepoint_debug_trace
from cellacdc.apps import (
    PreProcessParamsWidget,
    PreProcessRecipeDialogUtil,
    QBaseDialog,
    _CombineMetricsDialog,
)


@contextmanager
def disabled_while_nested(widget):
    """Disable the window of `widget` while a nested editor runs.

    `QBaseDialog.exec_` is not modal: without this, the calling dialog could
    be confirmed, edited or open a second editor while the editor is open,
    and the editor's result would be lost or applied to a changed state.
    The editor stays usable (`QBaseDialog.show` enables its own window).
    """
    window = widget.window()
    wasEnabled = window.isEnabled()
    if wasEnabled:
        window.setEnabled(False)
    try:
        yield
    finally:
        if wasEnabled:
            window.setEnabled(True)


class _CustomAnnotationPredictorGroupbox(QGroupBox):
    """Predictor settings of a custom annotation (measurements viewer). 
    
    `options` is a dict with optional keys 'acdc_df', 'SizeT', 'allChNames', 
    'isZstack' and 'isSegm3D' used to edit and test the equation.
    """
    def __init__(self, options, predictor=None, autoApply=False, parent=None):
        from .. import custom_annotation_predictors
        from cellacdc import timepoint_predictors
        self._predictors = timepoint_predictors
        self._annotPredictors = custom_annotation_predictors
        super().__init__('Predictor', parent)
        self.acdc_df = options.get('acdc_df')
        self.SizeT = options.get('SizeT')
        self.allChNames = list(options.get('allChNames', []))
        self.isZstack = options.get('isZstack', False)
        self.isSegm3D = options.get('isSegm3D', False)
        self.nameGetter = options.get('nameGetter', lambda: '')
        self.validateDependencies = options.get('validateDependencies')
        if predictor is None:
            predictor = {'type': timepoint_predictors.PREDICTOR_NONE}
        self.equation = predictor.get('equation', '')
        self.otherCells = predictor.get('other_cells')
        self.mainCellFilter = predictor.get('main_cell_filter')
        
        layout = QFormLayout(self)
        self.autoApplyCheckBox = widgets.Toggle(
            label_text='Auto apply on unvisited cells', parent=self
        )
        self.autoApplyCheckBox.setToolTip(
            'Evaluate the predictor on each cell when it is displayed for '
            'the first time.'
        )
        layout.addRow('', self.autoApplyCheckBox)
        
        self.predictorCombobox = QComboBox()
        self.predictorCombobox.addItem(
            'None (manual annotation)', timepoint_predictors.PREDICTOR_NONE
        )
        self.predictorCombobox.addItem(
            'Equation', timepoint_predictors.PREDICTOR_EQUATION
        )
        self.predictorCombobox.addItem(
            'Select own function (Python file)', 
            timepoint_predictors.PREDICTOR_FILE
        )
        self.predictorCombobox.setToolTip(
            'Assign one annotation to the entire cell history.\n'
            'The predictor must return booleans (True/False).'
        )
        layout.addRow('Predictor:', self.predictorCombobox)

        self.consolidationCombobox = QComboBox()
        for mode in custom_annotation_predictors.CONSOLIDATION_MODES:
            self.consolidationCombobox.addItem(mode.capitalize(), mode)
        self.consolidationCombobox.setToolTip(
            'All: every result is True (default).\n'
            'None: no result is True.\n'
            'Most: strictly more than half are True; ties are False.\n'
            'A single boolean is applied to all rows before consolidation.'
        )
        layout.addRow('Consolidation:', self.consolidationCombobox)
        
        self.equationWidget = QWidget()
        equationLayout = QHBoxLayout(self.equationWidget)
        equationLayout.setContentsMargins(0, 0, 0, 0)
        self.equationLineEdit = QLineEdit(self.equation)
        self.equationLineEdit.setReadOnly(True)
        self.equationLineEdit.setPlaceholderText('No equation set')
        self.editEquationButton = widgets.calcPushButton('Edit equation...')
        equationLayout.addWidget(self.equationLineEdit)
        equationLayout.addWidget(self.editEquationButton)
        layout.addRow('', self.equationWidget)
        
        self.fileWidget = QWidget()
        fileLayout = QGridLayout(self.fileWidget)
        fileLayout.setContentsMargins(0, 0, 0, 0)
        self.fileLineEdit = QLineEdit(predictor.get('filepath', ''))
        self.fileLineEdit.setPlaceholderText('Python file with predictor(df)')
        self.browseFileButton = widgets.browseFileButton(
            'Browse...', ext={'Python': '.py'}, 
            title='Select Python file with predictor(df)'
        )
        self.saveTemplateButton = widgets.savePushButton('Save template...')
        fileInfo = QLabel(html_utils.paragraph(
            'The file must define <code>predictor(df)</code>. '
            '<code>df</code> is the table of <b>one cell</b> (one row per '
            'frame, with <code>frame_i</code> and <code>Cell_ID</code> '
            'columns). Return <b>one boolean per row</b> or a single boolean. '
            'All/None/Most consolidates the results into one annotation '
            'for the entire cell history.<br>'
            'Optionally, add the keyword argument <code>entire_df</code> '
            '(<code>predictor(df, entire_df=None)</code>) to also receive '
            'the table of all cells.', font_size='11px'
        ))
        fileInfo.setWordWrap(True)
        fileLayout.addWidget(self.fileLineEdit, 0, 0)
        fileLayout.addWidget(self.browseFileButton, 0, 1)
        fileLayout.addWidget(self.saveTemplateButton, 0, 2)
        fileLayout.addWidget(fileInfo, 1, 0, 1, 3)
        layout.addRow('', self.fileWidget)
        
        testLayout = QHBoxLayout()
        self.statusLabel = QLabel()
        self.statusLabel.setWordWrap(True)
        self.testButton = widgets.calcPushButton('Test predictor')
        testLayout.addWidget(self.statusLabel, stretch=1)
        testLayout.addWidget(self.testButton)
        layout.addRow(testLayout)

        self.setPredictor(predictor, autoApply)

        self.predictorCombobox.currentIndexChanged.connect(self.predictorChanged)
        self.consolidationCombobox.currentIndexChanged.connect(self.statusLabel.clear)
        self.editEquationButton.clicked.connect(self.editEquation)
        self.browseFileButton.sigPathSelected.connect(self.fileLineEdit.setText)
        self.saveTemplateButton.clicked.connect(self.saveTemplate)
        self.testButton.clicked.connect(self.testPredictor)

    def setPredictor(self, predictor=None, autoApply=False):
        """Show a saved `predictor` dict (None: manual annotation) and the
        auto-apply option (used at init and by 'Load annotation...')."""
        if predictor is None:
            predictor = {'type': self._predictors.PREDICTOR_NONE}
        self.equation = predictor.get('equation', '')
        self.otherCells = predictor.get('other_cells')
        self.mainCellFilter = predictor.get('main_cell_filter')
        self.equationLineEdit.setText(self.equation)
        self.fileLineEdit.setText(predictor.get('filepath', ''))
        self.consolidationCombobox.setCurrentIndex(max(
            self.consolidationCombobox.findData(
                predictor.get('consolidation', 'all')
            ), 0
        ))
        index = self.predictorCombobox.findData(predictor.get('type'))
        self.predictorCombobox.setCurrentIndex(max(index, 0))
        self.predictorChanged()
        self.autoApplyCheckBox.setChecked(bool(autoApply))

    def predictorType(self):
        return self.predictorCombobox.currentData()

    def predictorChanged(self, *args):
        kind = self.predictorType()
        self.equationWidget.setVisible(
            kind == self._predictors.PREDICTOR_EQUATION
        )
        self.fileWidget.setVisible(kind == self._predictors.PREDICTOR_FILE)
        self.testButton.setVisible(kind != self._predictors.PREDICTOR_NONE)
        self.consolidationCombobox.setEnabled(
            kind != self._predictors.PREDICTOR_NONE
        )
        self.statusLabel.clear()
    
    def editEquation(self):
        win = CustomAnnotationPredictorEquationDialog(
            self.allChNames, self.isZstack, self.isSegm3D, 
            acdc_df=self.acdc_df, name=self.nameGetter(),
            equation=self.equation, otherCells=self.otherCells,
            mainCellFilter=self.mainCellFilter, SizeT=self.SizeT, parent=self,
            consolidation=self.consolidationCombobox.currentData()
        )
        if self.validateDependencies is not None:
            win.validatePredictorDependencies = (
                lambda predictor: self.validateDependencies(
                    self.nameGetter(), predictor
                )
            )
        with disabled_while_nested(self):
            win.exec_()
        # Deferred (its result is read below): editors are children of
        # dialogs that stay alive and keep references to the table
        win.deleteLater()
        if win.cancel:
            return
        self.equation = win.equation
        self.otherCells = win.otherCells
        self.mainCellFilter = win.mainCellFilter
        self.equationLineEdit.setText(self.equation)
        self.statusLabel.clear()
    
    def saveTemplate(self):
        filepath = QFileDialog.getSaveFileName(
            self, 'Save predictor template', 'custom_annotation_predictor.py',
            'Python (*.py)'
        )[0]
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(self._annotPredictors.PREDICTOR_FILE_TEMPLATE)
        except OSError as err:
            self.statusLabel.setText(html_utils.paragraph(
                f'<font color="red">{html.escape(str(err))}</font>'
            ))
            return
        self.fileLineEdit.setText(filepath)
    
    def predictor(self):
        kind = self.predictorType()
        if kind == self._predictors.PREDICTOR_EQUATION:
            predictor = {
                'type': kind, 'equation': self.equation,
                'consolidation': self.consolidationCombobox.currentData()
            }
            if self.otherCells is not None:
                predictor['other_cells'] = dict(self.otherCells)
            if self.mainCellFilter is not None:
                predictor['main_cell_filter'] = dict(self.mainCellFilter)
            return predictor
        if kind == self._predictors.PREDICTOR_FILE:
            return {
                'type': kind, 'filepath': self.fileLineEdit.text().strip(),
                'consolidation': self.consolidationCombobox.currentData()
            }
        return {'type': kind}
    
    def validatePredictor(self):
        Predictor = self._annotPredictors.CustomAnnotationPredictor
        predictor = self.predictor()
        Predictor.validate(predictor)
        if self.validateDependencies is not None:
            self.validateDependencies(self.nameGetter(), predictor)
        if predictor['type'] == self._predictors.PREDICTOR_FILE:
            Predictor.load_function(predictor['filepath'])
        return predictor
    
    def testPredictor(self):
        Predictor = self._annotPredictors.CustomAnnotationPredictor
        try:
            predictor = self.validatePredictor()
            if self.acdc_df is None or self.acdc_df.empty:
                self.statusLabel.setText(
                    'Predictor is valid. Load measurements to test it on data.'
                )
                return True
            annotated = Predictor.predict(
                self.acdc_df, predictor, SizeT=self.SizeT
            )
        except Exception as err:
            traceback.print_exc()
            self.statusLabel.setText(html_utils.paragraph(
                f'<font color="red">{html.escape(str(err))}</font>'
            ))
            return False
        numCells = self.acdc_df.index.get_level_values('Cell_ID').nunique()
        numFrames = sum(len(frames) for frames in annotated.values())
        self.statusLabel.setText(
            f'Annotated {numFrames} frame(s) of {len(annotated)} '
            f'of {numCells} cells.'
        )
        return True


class TimepointPredictorEquationDialog(_CombineMetricsDialog):
    """Equation editor (as in `CombineMetricsEquationDialog`) to predict
    the time points of each cell from the columns of `acdc_df`.

    A saved other-cell relation source (mother, sister or daughter) is
    converted into the equivalent same-frame candidate filter. When the
    loaded data lack the lineage tree or budding annotations of that
    source, the dialog still opens: the saved relation is kept unchanged
    as a read-only "<source> (annotations missing)" entry with a warning,
    and Test reports the prediction error."""
    def __init__(
            self, allChNames, isZstack, isSegm3D, acdc_df=None, name='',
            equation='', mode='first_true', parent=None,
            otherCells=None, mainCellFilter=None, SizeT=None,
        ):
        from cellacdc import timepoint_predictors
        self._predictors = timepoint_predictors
        self.SizeT = SizeT
        self.includeOtherMeasurements = True
        super().__init__(
            allChNames, isZstack, isSegm3D, parent=parent, acdc_df=acdc_df,
            allowCellAggregations=True
        )
        self.formulaEditWidget.setLiveValidation(False)
        self.setWindowTitle(f'Time point predictor equation - {name}')
        self.equation = equation
        self.mode = mode
        self._otherCells = (
            timepoint_predictors.CellFilter.default_other()
            if otherCells is None else
            timepoint_predictors.CellFilter.validate_other(otherCells)
        )
        self.otherCellFilterPresets = timepoint_predictors.CellFilter.other_presets(
            self.acdcDf
        )
        # Saved relation source (mother/sister/daughter) whose lineage tree
        # or budding annotations are missing in the loaded data: it is kept
        # unchanged (read-only, with a warning) instead of being converted.
        self._unavailableOtherSource = None
        source = self._otherCells['source']
        if source == timepoint_predictors.SOURCE_SAME_FRAME:
            pass
        elif source not in self.otherCellFilterPresets:
            self._unavailableOtherSource = source
        else:
            sourceCondition = self.otherCellFilterPresets[source][0]
            condition = self._otherCells['condition'].strip()
            self._otherCells['condition'] = (
                f'({sourceCondition}) & ({condition})'
                if condition else sourceCondition
            )
            self._otherCells['source'] = timepoint_predictors.SOURCE_SAME_FRAME
        self._mainCellFilter = (
            None if mainCellFilter is None else
            timepoint_predictors.CellFilter.validate_main(mainCellFilter)
        )
        self.newColNameLineEdit.setText(name)
        self.newColNameLineEdit.setReadOnly(True)
        for label in self.findChildren(QLabel):
            if label.text() == 'New measurement name:':
                label.setText('Time point identifier:')
        self.formulaEditWidget._edit.setPlaceholderText(
            'e.g. cell_area_pxl > 0.5 * MAX(cell_area_pxl) or '
            'cell_cycle_stage == "S"'
        )
        
        self.modeCombobox = QComboBox()
        modes = list(timepoint_predictors.EQUATION_MODES)
        for key in modes:
            self.modeCombobox.addItem(
                timepoint_predictors.EQUATION_MODES[key], key
            )
        self.modeCombobox.setCurrentIndex(
            modes.index(mode) if mode in modes else 0
        )
        self.modeCombobox.setToolTip(
            "The equation is evaluated for every frame of each cell. This\n"
            "rule converts the per-frame results into the cell's time points."
        )
        modeLayout = QHBoxLayout()
        modeLayout.addWidget(QLabel('Time point(s) of each cell:'))
        modeLayout.addWidget(self.modeCombobox)
        modeLayout.addStretch(1)
        
        functionsLayout = QVBoxLayout()
        cellFunctionsLayout = QHBoxLayout()
        cellFunctionsLayout.addWidget(QLabel('Per-cell functions:'))
        self.cellFunctionButtons = []
        for funcName, (_, description) in (
                timepoint_predictors.CELL_FUNCTIONS.items()):
            if funcName in ('any', 'all', 'ANY', 'ALL'):
                continue
            button = widgets.PushButton(funcName)
            button.setToolTip(qutils.wrap_tooltip(description))
            button.clicked.connect(
                partial(self.insertEquationText, f'{funcName}(')
            )
            cellFunctionsLayout.addWidget(button)
            self.cellFunctionButtons.append(button)
        cellFunctionsLayout.addStretch(1)

        functionsLayout.addLayout(cellFunctionsLayout)

        self.mainCellGroup = QGroupBox('Main-cell filter')
        self.mainCellGroup.setToolTip(
            'Limit which cells are evaluated by the predictor equation.'
        )
        self.mainCellGroup.setCheckable(True)
        self.mainCellGroup.setChecked(mainCellFilter is not None)
        mainCellLayout = QGridLayout(self.mainCellGroup)
        self.mainCellFilterPresets = {
            'All cells': '',
        }
        if self.SizeT is not None:
            self.mainCellFilterPresets.update({
                'Present at the last frame': 'LAST(frame_i) == SizeT - 1',
                'Not present at the last frame': 'LAST(frame_i) != SizeT - 1',
                'Present at the first frame': 'FIRST(frame_i) == 0',
                'Not present at the first frame': 'FIRST(frame_i) != 0',
                'Present throughout the video': (
                    '(FIRST(frame_i) == 0) & '
                    '(LAST(frame_i) == SizeT - 1) & '
                    '(NUNIQUE(frame_i) == SizeT)'
                ),
            })
        if self.acdcDf is not None:
            if 'relationship' in self.acdcDf:
                self.mainCellFilterPresets.update({
                    'Mother cells (budding)': 'relationship == "mother"',
                    'Bud cells (budding)': 'relationship == "bud"',
                })
            if 'cell_cycle_stage' in self.acdcDf:
                self.mainCellFilterPresets.update({
                    'Cells in G1 (budding)': 'cell_cycle_stage == "G1"',
                    'Cells in S (budding)': 'cell_cycle_stage == "S"',
                })
            if 'parent_ID_tree' in self.acdcDf:
                # Only positive IDs are parents: roots have -1, rows that
                # were not annotated 0 (`store_data` fill value) or NaN
                self.mainCellFilterPresets.update({
                    'Cells with parent': 'parent_ID_tree > 0',
                })
        mainCellCondition = (
            '' if self._mainCellFilter is None
            else self._mainCellFilter['condition']
        )
        self._mainCellCustomCondition = mainCellCondition
        mainCellMode = (
            'any' if self._mainCellFilter is None
            else self._mainCellFilter['condition_mode']
        )
        # Presets keep the cells where their condition is true in any
        # frame; the mode chosen under "Custom equation" is restored there
        self._mainCellCustomMode = mainCellMode
        self.mainCellFilterCombobox = QComboBox()
        mainCellPreset = 'custom'
        for label, condition in self.mainCellFilterPresets.items():
            self.mainCellFilterCombobox.addItem(label, label)
            # Without a condition the mode does not matter
            if (condition == mainCellCondition
                    and (mainCellMode == 'any' or not condition)):
                mainCellPreset = label
        self.mainCellFilterCombobox.addItem('Custom equation', 'custom')
        self.mainCellFilterCombobox.setCurrentIndex(
            self.mainCellFilterCombobox.findData(mainCellPreset)
        )
        self.mainCellConditionEdit = QLineEdit(
            mainCellCondition
        )
        filterTooltip = (
            'Express "if" as a boolean condition, e.g. area > 5. Combine conditions\n'
            'with & (and), | (or), and ~ (not), with parentheses around each\n'
            'comparison: (area > 5) & (cell_cycle_stage == "S"). Do not use Python\n'
            'if/else statements or and/or. SHIFT(area, 1) is the previous observed\n'
            'value of this cell (not necessarily the previous video frame).'
        )
        self.mainCellConditionEdit.setToolTip(filterTooltip)
        self.formulaEditWidget._edit.setToolTip(filterTooltip)
        self.mainCellConditionButton = widgets.editPushButton('Edit filter...')
        self.mainCellMatchCombobox = QComboBox()
        for key in ('any', 'all'):
            self.mainCellMatchCombobox.addItem(
                timepoint_predictors.FILTER_FRAME_MODES[key], key
            )
        self.mainCellMatchCombobox.setCurrentIndex(
            self.mainCellMatchCombobox.findData(
                mainCellMode if mainCellPreset == 'custom' else 'any'
            )
        )
        mainCellLayout.addWidget(QLabel('Select cells:'), 0, 0)
        mainCellLayout.addWidget(self.mainCellFilterCombobox, 0, 1, 1, 2)
        mainCellLayout.addWidget(QLabel('Equation:'), 1, 0)
        mainCellLayout.addWidget(self.mainCellConditionEdit, 1, 1)
        mainCellLayout.addWidget(self.mainCellConditionButton, 1, 2)
        mainCellLayout.addWidget(QLabel('Keep cells where true in:'), 2, 0)
        mainCellLayout.addWidget(self.mainCellMatchCombobox, 2, 1, 1, 2)

        self.otherCellsGroup = QGroupBox('Other-cell measurements')
        self.otherCellsGroup.setToolTip(
            'Configure which other-cell measurements are available in the '
            'second tree.'
        )
        self.otherCellsGroup.setCheckable(True)
        self.otherCellsGroup.setChecked(otherCells is not None)
        otherCellsLayout = QGridLayout(self.otherCellsGroup)
        self._otherCellsCustomCondition = self._otherCells['condition']
        self._otherCellsCustomRange = (
            self._otherCells['range_start'], self._otherCells['range_end']
        )
        self.otherCellsSourceCombobox = QComboBox()
        for key, label in (
                (timepoint_predictors.SOURCE_SAME_FRAME, 'Other cells'),
                (timepoint_predictors.SOURCE_MOTHER, 'Mother cell'),
                (timepoint_predictors.SOURCE_SISTER, 'Sister cell(s)'),
                (timepoint_predictors.SOURCE_DAUGHTER, 'Daughter cell(s)')
            ):
            if key in self.otherCellFilterPresets:
                self.otherCellsSourceCombobox.addItem(label, key)
        missingSourceLabel = ''
        if self._unavailableOtherSource is not None:
            missingSourceLabel = timepoint_predictors.SOURCE_CELLS[
                self._unavailableOtherSource
            ]
            self.otherCellsSourceCombobox.addItem(
                f'{missingSourceLabel} (annotations missing)',
                self._unavailableOtherSource
            )
        self.otherCellsSourceCombobox.addItem(
            'Custom equation', timepoint_predictors.SOURCE_CUSTOM
        )
        otherCellsPreset = timepoint_predictors.SOURCE_CUSTOM
        for key, (condition, start, end) in self.otherCellFilterPresets.items():
            if (condition == self._otherCells['condition']
                    and (start, end) == self._otherCellsCustomRange):
                otherCellsPreset = key
        if self._unavailableOtherSource is not None:
            otherCellsPreset = self._unavailableOtherSource
        self.otherCellsMissingSourceLabel = QLabel(html_utils.paragraph(
            '<font color="red">The loaded measurements lack the lineage tree '
            '(<code>parent_ID_tree</code>) or budding '
            '(<code>relative_ID</code>, <code>relationship</code>) '
            f'annotations required by "{html.escape(missingSourceLabel)}". '
            'The saved relation is kept unchanged, but it cannot be '
            'evaluated on these data. Select another source or "Custom '
            'equation" to replace it.</font>'
        ))
        self.otherCellsMissingSourceLabel.setWordWrap(True)
        self.otherCellsMissingSourceLabel.setVisible(
            otherCellsPreset == self._unavailableOtherSource
        )
        self.otherCellsSourceCombobox.setCurrentIndex(
            self.otherCellsSourceCombobox.findData(otherCellsPreset)
        )
        self.otherCellsConditionEdit = QLineEdit(self._otherCells['condition'])
        self.otherCellsConditionEdit.setToolTip(
            filterTooltip + '\nCandidate inputs use normal names; this_ '
            'inputs refer to the enquiring\ncell at frame t, even when '
            'candidates come from another frame.'
        )
        self.otherCellsConditionButton = widgets.editPushButton('Edit filter...')
        self.otherCellsMatchCombobox = QComboBox()
        for key, label in timepoint_predictors.FILTER_FRAME_MODES.items():
            self.otherCellsMatchCombobox.addItem(label, key)
        self.otherCellsMatchCombobox.setCurrentIndex(
            self.otherCellsMatchCombobox.findData(
                self._otherCells['condition_mode']
            )
        )
        self.otherCellsRangeStartEdit = QLineEdit(
            self._otherCells['range_start']
        )
        self.otherCellsRangeEndEdit = QLineEdit(self._otherCells['range_end'])
        for edit in (
                self.otherCellsRangeStartEdit, self.otherCellsRangeEndEdit
            ):
            edit.setToolTip(
                'Inclusive frame bounds relative to t, '
                'the frame being predicted (e.g. t - 2 to\n'
                't + 1). Shift both bounds to t - 1 for '
                'the previous frame (symmetric mother),\n'
                'or use t for budding relatives in the '
                'same frame. Unlike SHIFT(column, 1),\n'
                'this selects an exact video frame; missing frames are not padded.'
            )
        otherCellsLayout.addWidget(QLabel('Select cells:'), 0, 0)
        otherCellsLayout.addWidget(self.otherCellsSourceCombobox, 0, 1, 1, 2)
        otherCellsLayout.addWidget(QLabel('Equation:'), 1, 0)
        otherCellsLayout.addWidget(self.otherCellsConditionEdit, 1, 1)
        otherCellsLayout.addWidget(self.otherCellsConditionButton, 1, 2)
        otherCellsLayout.addWidget(QLabel('Keep candidates:'), 2, 0)
        otherCellsLayout.addWidget(self.otherCellsMatchCombobox, 2, 1)
        otherCellsLayout.addWidget(QLabel('Frame window:'), 3, 0)
        rangeLayout = QHBoxLayout()
        rangeLayout.addWidget(self.otherCellsRangeStartEdit)
        rangeLayout.addWidget(QLabel('to'))
        rangeLayout.addWidget(self.otherCellsRangeEndEdit)
        otherCellsLayout.addLayout(rangeLayout, 3, 1, 1, 2)
        otherCellsLayout.addWidget(self.otherCellsMissingSourceLabel, 4, 0, 1, 3)
        mainLayout = self.layout()
        mainLayout.insertWidget(mainLayout.count()-3, self.mainCellGroup)
        mainLayout.insertWidget(mainLayout.count()-3, self.otherCellsGroup)
        mainLayout.insertLayout(mainLayout.count()-3, functionsLayout)
        self.otherCellsGroup.toggled.connect(self.updateOtherCellsControls)
        self.otherCellsGroup.toggled.connect(self.equationChanged)
        self.mainCellGroup.toggled.connect(self.updateMainCellControls)
        self.mainCellGroup.toggled.connect(self.equationChanged)
        self.mainCellMatchCombobox.currentIndexChanged.connect(
            self.mainCellModeChanged
        )
        self.mainCellConditionEdit.textChanged.connect(
            self.mainCellConditionChanged
        )
        self.mainCellFilterCombobox.currentIndexChanged.connect(
            self.mainCellPresetChanged
        )
        self.otherCellsSourceCombobox.currentIndexChanged.connect(
            self.otherCellsPresetChanged
        )
        self.otherCellsMatchCombobox.currentIndexChanged.connect(
            self.equationChanged
        )
        self.otherCellsConditionEdit.textChanged.connect(
            self.otherCellsConditionChanged
        )
        self.otherCellsRangeStartEdit.textChanged.connect(self.otherCellsRangeChanged)
        self.otherCellsRangeEndEdit.textChanged.connect(self.otherCellsRangeChanged)
        self.otherCellsConditionButton.clicked.connect(
            self.editOtherCellsCondition
        )
        self.mainCellConditionButton.clicked.connect(
            self.editMainCellCondition
        )
        self.updateMainCellControls(self.mainCellGroup.isChecked())
        self.updateOtherCellsControls(self.otherCellsGroup.isChecked())

        mainLayout.insertLayout(mainLayout.count()-3, modeLayout)
        self.modeCombobox.currentIndexChanged.connect(self.equationChanged)
        self.otherCells = self.currentOtherCells()
        if equation:
            self.formulaEditWidget.setText(equation)

    def updateOtherCellsControls(self, enabled):
        preset = self.otherCellsSourceCombobox.currentData()
        custom = preset == 'custom'
        self.otherCellsMissingSourceLabel.setVisible(
            self._unavailableOtherSource is not None
            and preset == self._unavailableOtherSource
        )
        self.otherCellsSourceCombobox.setEnabled(enabled)
        for widget in (
                self.otherCellsConditionEdit,
                self.otherCellsConditionButton,
                self.otherCellsMatchCombobox,
                self.otherCellsRangeStartEdit,
                self.otherCellsRangeEndEdit
            ):
            widget.setEnabled(enabled and custom)
        self.otherMeasurementsTreeWidget.setEnabled(enabled)

    def updateMainCellControls(self, enabled):
        custom = self.mainCellFilterCombobox.currentData() == 'custom'
        self.mainCellFilterCombobox.setEnabled(enabled)
        for widget in (
                self.mainCellConditionEdit,
                self.mainCellConditionButton
            ):
            widget.setEnabled(enabled and custom)
        self.mainCellMatchCombobox.setEnabled(enabled and custom)

    def mainCellConditionChanged(self, text):
        if self.mainCellFilterCombobox.currentData() == 'custom':
            self._mainCellCustomCondition = text
        self.equationChanged()

    def mainCellModeChanged(self, index):
        if self.mainCellFilterCombobox.currentData() == 'custom':
            self._mainCellCustomMode = self.mainCellMatchCombobox.currentData()
        self.equationChanged()

    def otherCellsConditionChanged(self, text):
        if self.otherCellsSourceCombobox.currentData() == 'custom':
            self._otherCellsCustomCondition = text
        self.equationChanged()

    def otherCellsRangeChanged(self, text):
        if self.otherCellsSourceCombobox.currentData() == 'custom':
            self._otherCellsCustomRange = (
                self.otherCellsRangeStartEdit.text(),
                self.otherCellsRangeEndEdit.text()
            )
        self.equationChanged()

    def mainCellPresetChanged(self, index):
        preset = self.mainCellFilterCombobox.currentData()
        custom = preset == 'custom'
        condition = (
            self._mainCellCustomCondition if custom
            else self.mainCellFilterPresets[preset]
        )
        # Presets always keep the cells where they are true in any frame
        mode = self._mainCellCustomMode if custom else 'any'
        self.mainCellMatchCombobox.blockSignals(True)
        self.mainCellMatchCombobox.setCurrentIndex(
            self.mainCellMatchCombobox.findData(mode)
        )
        self.mainCellMatchCombobox.blockSignals(False)
        self.mainCellConditionEdit.setText(condition)
        self.updateMainCellControls(self.mainCellGroup.isChecked())
        self.equationChanged()

    def otherCellsPresetChanged(self, index):
        preset = self.otherCellsSourceCombobox.currentData()
        if preset == 'custom':
            condition = self._otherCellsCustomCondition
            start, end = self._otherCellsCustomRange
        elif preset == self._unavailableOtherSource:
            condition = self._otherCells['condition']
            start = self._otherCells['range_start']
            end = self._otherCells['range_end']
        else:
            condition, start, end = self.otherCellFilterPresets[preset]
        self.otherCellsConditionEdit.setText(condition)
        for edit, value in (
                (self.otherCellsRangeStartEdit, start),
                (self.otherCellsRangeEndEdit, end)
            ):
            edit.blockSignals(True)
            edit.setText(value)
            edit.blockSignals(False)
        self.updateOtherCellsControls(self.otherCellsGroup.isChecked())
        self.equationChanged()

    def currentOtherCells(self):
        if not self.otherCellsGroup.isChecked():
            return None
        preset = self.otherCellsSourceCombobox.currentData()
        if (self._unavailableOtherSource is not None
                and preset == self._unavailableOtherSource):
            # Keep the saved relation (and its source) unchanged
            return self._predictors.CellFilter.validate_other(
                dict(self._otherCells)
            )
        settings = {
            'source': self._predictors.SOURCE_SAME_FRAME,
            'condition': self.otherCellsConditionEdit.text().strip(),
            'condition_mode': self.otherCellsMatchCombobox.currentData(),
            'range_start': self.otherCellsRangeStartEdit.text().strip(),
            'range_end': self.otherCellsRangeEndEdit.text().strip(),
        }
        return self._predictors.CellFilter.validate_other(settings)

    def currentMainCellFilter(self):
        if not self.mainCellGroup.isChecked():
            return None
        preset = self.mainCellFilterCombobox.currentData()
        custom = preset == 'custom'
        settings = {
            'condition': (
                self.mainCellConditionEdit.text().strip() if custom
                else self.mainCellFilterPresets[preset]
            ),
            'condition_mode': (
                self.mainCellMatchCombobox.currentData() if custom else 'any'
            ),
        }
        return self._predictors.CellFilter.validate_main(settings)

    def editOtherCellsCondition(self, checked=False):
        # The candidate filter does not depend on the frame window, so the
        # editor opens even while the window is incomplete or invalid
        dialog = _TimepointCellFilterEquationDialog(
            self.allChNames, self.isZstack, self.isSegm3D,
            acdc_df=self.acdcDf, SizeT=self.SizeT,
            condition=self.otherCellsConditionEdit.text().strip(),
            includeEnquiringCell=True,
            parent=self
        )
        dialog.formulaEditWidget._edit.setToolTip(
            self.otherCellsConditionEdit.toolTip()
        )
        with disabled_while_nested(self):
            dialog.exec_()
        # It holds a flattened copy of the table (`acdcDfs`)
        dialog.deleteLater()
        if not dialog.cancel:
            self.otherCellsConditionEdit.setText(dialog.equation)

    def editMainCellCondition(self, checked=False):
        try:
            otherCells = self.currentOtherCells()
        except ValueError as err:
            self._showTestError(f'Invalid other-cell settings: {err}')
            return
        dialog = _TimepointCellFilterEquationDialog(
            self.allChNames, self.isZstack, self.isSegm3D,
            acdc_df=self.acdcDf, SizeT=self.SizeT,
            condition=self.mainCellConditionEdit.text(),
            otherCells=otherCells,
            parent=self
        )
        dialog.formulaEditWidget._edit.setToolTip(
            self.mainCellConditionEdit.toolTip()
        )
        with disabled_while_nested(self):
            dialog.exec_()
        dialog.deleteLater()
        if not dialog.cancel:
            self.mainCellConditionEdit.setText(dialog.equation)
    
    def currentMode(self):
        return self.modeCombobox.currentData()
    
    @timepoint_debug_trace
    def validateEquation(self, formula):
        try:
            if not formula.strip():
                return False, 'Equation cannot be empty'
            # frame_i and Cell_ID are index levels (not listed in the
            # trees) but are valid per-cell inputs, also of the other cells
            # (columns of CellTable.flatten); SizeT is a constant.
            indexInputs = ('frame_i', 'Cell_ID')
            usesIndexInputs = bool(
                {*indexInputs, *(f'{name}_other' for name in indexInputs)}
                & self._predictors.CellEquation.names(formula)
            )
            if (
                    not self.formulaEditWidget.referencedVariableNames()
                    and not usesIndexInputs
                ):
                return False, 'No measurement inputs used'
            settings = self.currentOtherCells()
            if self._predictors._OTHER_FUNCTION_PATTERN.search(formula):
                if settings is None:
                    return False, (
                        'Enable Other-cell measurements to use *_OTHER(...).'
                    )
            otherMeasurementCalls = list(
                self._predictors._OTHER_MEASUREMENT_PATTERN.finditer(formula)
            )
            otherInputNames = {
                name for name in self._predictors.CellEquation.names(formula)
                if name.endswith('_other')
            }
            if otherInputNames and settings is None:
                return False, (
                    'Enable Other-cell measurements to use _other inputs.'
                )
            available = (
                self._predictors.CellTable.flatten(self.acdcDf).columns
                if self.acdcDf is not None else ()
            )
            for otherName in otherInputNames:
                if otherName[:-len('_other')] in indexInputs:
                    continue
                if otherName not in self.otherMeasurementColumns:
                    return False, f'Unknown other-cell input "{otherName}".'
                column = self.otherMeasurementColumns[otherName]
                if self.acdcDf is not None and column not in available:
                    return False, f'Unknown input column "{column}".'
            legacyCalls = bool(
                self._predictors._OTHER_FUNCTION_PATTERN.search(formula)
            )
            if legacyCalls or otherMeasurementCalls:
                for match in self._predictors._OTHER_FUNCTION_PATTERN.finditer(formula):
                    column = match.group('column').strip('`')
                    if self.acdcDf is not None and column not in available:
                        return False, f'Unknown input column "{column}".'
            if self.acdcDf is not None and (legacyCalls or otherInputNames):
                # Real candidate values with their dtypes (yes/no inputs
                # with ~) and frame_i_other/Cell_ID_other, as in the
                # library and _TimepointCellFilterEquationDialog
                df, evaluated, alternatives = (
                    self._predictors.OtherCellMeasurements.prepare_inputs(
                        self.acdcDf, formula, settings, self.SizeT,
                        max_targets=10, return_alternatives=True
                    )
                )
                self.evaluateEquation(df, evaluated, alternatives=alternatives)
                return True, formula
            # Without a loaded table: placeholder values
            if legacyCalls or otherMeasurementCalls:
                formula = self._predictors._OTHER_FUNCTION_PATTERN.sub(
                    '1', formula
                )
                formula = self._predictors._OTHER_MEASUREMENT_PATTERN.sub(
                    '1', formula
                )
            df = self.getTestData(preview=True)
            if self.acdcDf is None:
                usedIndexInputs = (
                    {*indexInputs, *(f'{name}_other' for name in indexInputs)}
                    & self._predictors.CellEquation.names(formula)
                )
                for name in usedIndexInputs:
                    if name not in df and name not in df.index.names:
                        df[name] = 0
            self.evaluateEquation(df, formula)
        except Exception as err:
            return False, str(err)
        return True, formula
    
    def _showTestError(self, message):
        self.statusLabel.setText(message)
        self.testOutputDisplay.setPlainText(message)
        self.testOutputDisplay.setStyleSheet('border: 2px solid red')
    
    @timepoint_debug_trace
    def test_cb(self):
        if self.acdcDf is None:
            self.equationChanged()
            success, message = self.formulaEditWidget.checkValidity()
            if not success:
                self._showTestError(message)
                return
            self.statusLabel.setText(
                'Equation is valid. Load measurements to test its predictions.'
            )
            self.testOutputDisplay.setPlainText(
                'Equation validation only; prediction requires loaded measurements.'
            )
            self.testOutputDisplay.setStyleSheet('border: 2px solid green')
            self.okButton.setEnabled(True)
            return
        self.equationChanged()
        success, message = self.formulaEditWidget.checkValidity()
        if not success:
            self._showTestError(message)
            return
        try:
            timepoints = self._predictors.TimepointPredictor.from_equation(
                self.acdcDf, self.formulaEditWidget.text(), self.currentMode(),
                SizeT=getattr(self, 'SizeT', None),
                other_cells=self.currentOtherCells(),
                main_cell_filter=self.currentMainCellFilter()
            )
        except Exception as err:
            traceback.print_exc()
            self._showTestError(str(err))
            return
        withTimepoints = {ID: frames for ID, frames in timepoints.items() if frames}
        preview = pd.DataFrame({
            'Cell_ID': list(withTimepoints),
            'Predicted frame n.': [
                ', '.join(str(frame+1) for frame in frames)
                for frames in withTimepoints.values()
            ]
        }).head(20)
        self.testOutputDisplay.setHtml(preview.to_html(index=False))
        self.testOutputDisplay.setStyleSheet('border: 2px solid green')
        self.statusLabel.setText(
            f'Predicted time points for {len(withTimepoints)} of '
            f'{len(timepoints)} cells.'
        )
        self.okButton.setEnabled(True)
    
    def ok_cb(self):
        if not self.okButton.isEnabled():
            self.test_cb()
            return
        validator = getattr(self, 'validatePredictorDependencies', None)
        if validator is not None:
            predictor = {
                'type': self._predictors.PREDICTOR_EQUATION,
                'equation': self.formulaEditWidget.text(),
                'other_cells': self.currentOtherCells(),
                'main_cell_filter': self.currentMainCellFilter(),
            }
            try:
                validator(predictor)
            except ValueError as err:
                msg = widgets.myMessageBox(wrapText=False)
                msg.warning(
                    self, 'Invalid predictor',
                    html_utils.paragraph(html.escape(str(err)))
                )
                return
        self.equation = self.formulaEditWidget.text()
        self.mode = self.currentMode()
        self.otherCells = self.currentOtherCells()
        self.mainCellFilter = self.currentMainCellFilter()
        self.cancel = False
        self.close()
    
    def showHelp(self):
        functions = '<br>'.join(
            f'<code>{name}(x)</code>: {html.escape(description)}'
            for name, (_, description) 
            in self._predictors.CELL_FUNCTIONS.items()
        )
        txt = html_utils.paragraph(f"""
            The equation is evaluated <b>separately for each cell</b> on its 
            rows of the measurements table (one row per frame).<br><br>
            Use conditions (e.g., <code>cell_area_pxl &gt; 500</code>) with the 
            rules based on "true", or numerical expressions with the 
            "maximum"/"minimum" rules.<br><br>
            <b>Per-cell functions</b> are computed over all frames of the 
            cell (aggregates ignore NaNs; SHIFT/DIFF use the cell's
            observed rows) and can be combined with the other
            inputs, e.g. <code>cell_area_pxl &gt; 0.5 * MAX(cell_area_pxl)</code> 
            or <code>DIFF(cell_area_pxl) &gt; 50</code>. <code>any(...)</code>
            and <code>all(...)</code> reduce a condition over all frames of
            that cell.<br>{functions}<br><br>
            <b>Main-cell filter</b> limits which cells are evaluated by the
            predictor equation. Select a preset for cells present at the start,
            end, or throughout the video. Cell-cycle presets are available
            when their annotation columns are loaded.
            Preset settings are read-only. Select "Custom equation" to enable
            the equation field, editor and match mode.
            "Any" keeps a cell if the filter is true in
            at least one of its frames; "All" requires it to be true in every
            frame. Other-cell inputs use the configured Other-cell source and
            frame window. To keep cells that have daughters, select the Daughter
            preset under Other-cell measurements and use
            <code>COUNT(cell_area_pxl_other) &gt; 0</code> with "Any" in the
            main-cell filter. The predictor then evaluates all frames of each
            retained cell.<br><br>
            <b>Other-cell measurements</b> are listed in the second tree with
            an <code>_other</code> suffix. Aggregate one in the equation with
            the usual functions, e.g. <code>area &gt; MEAN(area_other)</code>.
            A direct input such as <code>area_other</code> uses the selected
            value. With several different values, the equation is evaluated
            once per distinct value (a warning is printed): a condition is
            true if any value makes it true, and the Max/Min rules use the
            extreme result. Use an aggregate such as
            <code>MEAN(area_other)</code> to combine the values into one
            instead. No selected rows produce 0 for other-cell inputs and
            aggregates; a direct input with only missing values also gives 0.
            The aggregation uses selected source cells in the configured
            inclusive frame window around t. Select a source preset or
            "Custom equation" to edit the displayed filter, match mode and
            frame window. Preset settings are read-only.
            Mother, sister and daughter sources need lineage tree or budding
            annotations; if the loaded data lack them, a saved source is
            shown as "(annotations missing)" and kept unchanged until
            you select another source.
            Candidate inputs use normal names; <code>this_</code>-prefixed
            names refer to the enquiring cell at t. Budding relatives use
            frame t; symmetric mothers use the shifted bounds
            <code>t - 1</code> to <code>t - 1</code>, and daughters
            <code>t + 1</code> to <code>t + 1</code>.
            <code>SHIFT(area, 1)</code> instead gives the previous observed
            measurement of a cell, not a relative's frame.<br><br>
            <b>If conditions:</b> write a boolean expression, e.g.
            <code>(area &gt; 5) &amp; (cell_cycle_stage == "S")</code>.
            Use <code>&amp;</code> (and), <code>|</code> (or),
            <code>~</code> (not) and parentheses, not Python
            <code>if/else</code> statements or <code>and/or</code>.<br><br>
            Double-click an input to add it to the equation.
            In the measurements viewer, time points are boolean
            <code>timepoint_&lt;name&gt;</code> inputs and custom annotations
            are 0/1 <code>&lt;name&gt;</code> inputs. Compare annotations with
            <code>== 1</code> for a boolean result. Prerequisite predictors
            run first; circular equation dependencies cannot be accepted.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, 'Time point predictor help', txt)


class CustomAnnotationPredictorEquationDialog(TimepointPredictorEquationDialog):
    """Equation editor of a custom annotation predictor. Same as 
    `TimepointPredictorEquationDialog` without the time point rule: the
    equation must return a boolean per frame (or a single boolean per cell)
    and All/None/Most consolidates it into one annotation for the cell."""
    def __init__(
            self, allChNames, isZstack, isSegm3D, acdc_df=None, name='',
            equation='', parent=None, otherCells=None, mainCellFilter=None,
            SizeT=None, consolidation='all'
        ):
        from .. import custom_annotation_predictors
        self._annotPredictors = custom_annotation_predictors
        self.consolidation = consolidation
        super().__init__(
            allChNames, isZstack, isSegm3D, acdc_df=acdc_df, name=name,
            equation=equation, parent=parent, otherCells=otherCells,
            mainCellFilter=mainCellFilter, SizeT=SizeT
        )
        self.setWindowTitle(f'Custom annotation predictor equation - {name}')
        for label in self.findChildren(QLabel):
            if label.text() == 'Time point identifier:':
                label.setText('Custom annotation:')
            elif label.text() == 'Time point(s) of each cell:':
                label.hide()
        self.modeCombobox.hide()
        self.formulaEditWidget._edit.setPlaceholderText(
            'e.g. cell_area_pxl > 500 or cell_cycle_stage == "S" '
            '(must return True/False)'
        )
    
    @timepoint_debug_trace
    def test_cb(self):
        if self.acdcDf is None:
            super().test_cb()
            return
        self.equationChanged()
        success, message = self.formulaEditWidget.checkValidity()
        if not success:
            self._showTestError(message)
            return
        Predictor = self._annotPredictors.CustomAnnotationPredictor
        try:
            annotated = Predictor.from_equation(
                self.acdcDf, self.formulaEditWidget.text(),
                SizeT=getattr(self, 'SizeT', None),
                other_cells=self.currentOtherCells(),
                main_cell_filter=self.currentMainCellFilter(),
                consolidation=self.consolidation
            )
        except Exception as err:
            traceback.print_exc()
            self._showTestError(str(err))
            return
        withFrames = {ID: frames for ID, frames in annotated.items() if frames}
        preview = pd.DataFrame({
            'Cell_ID': list(withFrames),
            'Annotated frame n.': [
                ', '.join(str(frame+1) for frame in frames)
                for frames in withFrames.values()
            ]
        }).head(20)
        self.testOutputDisplay.setHtml(preview.to_html(index=False))
        self.testOutputDisplay.setStyleSheet('border: 2px solid green')
        numFrames = sum(len(frames) for frames in withFrames.values())
        self.statusLabel.setText(
            f'Annotated {numFrames} frame(s) of {len(withFrames)} of '
            f'{len(annotated)} cells.'
        )
        self.okButton.setEnabled(True)
    
    def showHelp(self):
        functions = '<br>'.join(
            f'<code>{name}(x)</code>: {html.escape(description)}'
            for name, (_, description) 
            in self._predictors.CELL_FUNCTIONS.items()
        )
        txt = html_utils.paragraph(f"""
            The equation is evaluated <b>separately for each cell</b> on its 
            rows of the measurements table (one row per frame).<br><br>
            The result <b>must be boolean</b> (True/False), either one value 
            per frame (e.g., <code>cell_area_pxl &gt; 500</code>) or a single 
            value for the entire cell (e.g., 
            <code>MAX(cell_area_pxl) &gt; 500</code>). Results are consolidated
            for the <b>entire cell history</b>: <b>All</b> requires every
            result to be True; <b>None</b> requires no True results;
            <b>Most</b> requires strictly more than half to be True.
            A single boolean is broadcast before consolidation. Numerical results
            (including 0/1) are not allowed.<br><br>
            <b>Per-cell functions</b> are computed over all frames of the 
            cell (aggregates ignore NaNs; SHIFT/DIFF use the cell's
            observed rows):<br>{functions}<br><br>
            <b>Main-cell filter</b> limits which cells are evaluated, and 
            <b>Other-cell measurements</b> (<code>_other</code> suffix) work 
            as in the time point predictors. With multiple other-cell values,
            a frame's result is True if the equation is True for any of them,
            before All/None/Most consolidation across the cell history.
            <br><br>
            <b>If conditions:</b> use <code>&amp;</code> (and), 
            <code>|</code> (or), <code>~</code> (not) and parentheses, not 
            Python <code>if/else</code> statements or <code>and/or</code>.
            <br><br>
            Double-click an input to add it to the equation.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, 'Custom annotation predictor help', txt)


class _TimepointCellFilterEquationDialog(_CombineMetricsDialog):
    """Equation editor for the main-cell or other-cell filter of a
    predictor (and base of the conditional formatting condition editor).

    Test checks a preview and then the whole table; OK only re-tests when
    the OK button is not already enabled by a passing Test of the current
    equation (every edit disables it again).
    """
    # Offer the 'Other-cell measurements' tree (*_other inputs) in the
    # main-cell filter. Subclasses evaluated without other cells disable it.
    OTHER_CELL_INPUTS = True

    def __init__(
            self, allChNames, isZstack, isSegm3D, acdc_df=None, SizeT=None,
            condition='', includeEnquiringCell=False, parent=None,
            otherCells=None
        ):
        from cellacdc import timepoint_predictors
        self._predictors = timepoint_predictors
        self.SizeT = SizeT
        self.includeEnquiringCell = includeEnquiringCell
        self.otherCells = otherCells
        # Must be set before super().__init__, which builds the trees
        self.includeOtherMeasurements = (
            self.OTHER_CELL_INPUTS and not includeEnquiringCell
        )
        if includeEnquiringCell:
            df = (
                timepoint_predictors.CellTable.flatten(acdc_df)
                if acdc_df is not None else
                pd.DataFrame(columns=['frame_i', 'Cell_ID'])
            )
            super().__init__(
                allChNames, isZstack, isSegm3D, parent=parent,
                acdcDfs={'Candidate cell': df, 'Enquiring cell': df},
                allowCellAggregations=True
            )
        else:
            super().__init__(
                allChNames, isZstack, isSegm3D, parent=parent,
                acdc_df=acdc_df, allowCellAggregations=True
            )
        self.formulaEditWidget.setLiveValidation(False)
        self.setWindowTitle(
            'Other-cell filter equation' if includeEnquiringCell
            else 'Main-cell filter equation'
        )
        self.newColNameLineEdit.setText('Cell filter')
        self.newColNameLineEdit.setReadOnly(True)
        for label in self.findChildren(QLabel):
            if label.text() == 'New measurement name:':
                label.setText('Filter equation:')
        self.equation = condition
        functionsLayout = QHBoxLayout()
        functionsLayout.addWidget(QLabel('Per-cell functions:'))
        for name, (_, description) in timepoint_predictors.CELL_FUNCTIONS.items():
            if name in ('any', 'all', 'ANY', 'ALL'):
                continue
            button = widgets.PushButton(name)
            button.setToolTip(qutils.wrap_tooltip(description))
            button.clicked.connect(
                partial(self.insertEquationText, f'{name}(')
            )
            functionsLayout.addWidget(button)
        functionsLayout.addStretch(1)
        self.layout().insertLayout(self.layout().count()-3, functionsLayout)
        self.layout().insertWidget(0, QLabel(html_utils.paragraph(
            'Return bool or 0/1, either a scalar or one value per frame.'
            + (
                ' Candidate inputs use their measurement names; enquiring-cell '
                'inputs are prefixed with <code>this_</code>.'
                if includeEnquiringCell else
                ' The equation is evaluated on each target cell across its '
                'frames. Other-cell inputs use the source and frame window '
                'configured in the predictor. With Daughter cells selected, '
                '<code>COUNT(cell_area_pxl_other) &gt; 0</code> and match '
                '"Any" retain cells with observed daughters, without limiting '
                'the predictor to their division frame.'
            )
        )))
        self.formulaEditWidget.setText(condition)

    def addTreeItems(
            self, parentItem, itemsText, isCol=False, isChannelLess=False,
            index=None
        ):
        start = parentItem.childCount()
        super().addTreeItems(
            parentItem, itemsText, isCol=isCol,
            isChannelLess=isChannelLess, index=index
        )
        if not self.includeEnquiringCell or index is None:
            return
        for row in range(start, parentItem.childCount()):
            item = parentItem.child(row)
            if not getattr(item, 'isCol', False):
                continue
            oldName = item.variableName
            name = item.text(0)
            item.variableName = f'this_{name}' if index == 1 else name
            item.setData(
                0, widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE,
                item.variableName
            )
            self.variableNames[self.variableNames.index(oldName)] = (
                item.variableName
            )

    def getTestData(self, preview=False):
        if not self.includeEnquiringCell:
            if self.acdcDf is not None:
                return self.acdcDf.copy()
            return super().getTestData(preview=preview)
        candidate, enquiring = self.acdcDfs.values()
        candidate = (
            candidate.head(10) if preview else candidate
        ).reset_index(drop=True).copy()
        enquiring = (
            enquiring.head(10) if preview else enquiring
        ).reset_index(drop=True)
        if candidate.empty:
            candidate = pd.DataFrame(index=[0])
            enquiring = pd.DataFrame(index=[0])
            for name in self.formulaEditWidget.referencedVariableNames():
                variable = (
                    name[5:] if name.startswith('this_') else name
                )
                candidate[variable] = 1
                if name.startswith('this_'):
                    enquiring[variable] = 1
        # Both tables share the same RangeIndex: add all this_ columns at
        # once (column-wise inserts fragment the frame)
        return pd.concat([candidate, enquiring.add_prefix('this_')], axis=1)

    @timepoint_debug_trace
    def evaluateEquation(
            self, df, formula, *, enforce_dtype_output=None, preview=False
        ):
        if not self.includeEnquiringCell:
            if self.acdcDf is None:
                otherNames = [
                    name for name in self._predictors.CellEquation.names(formula)
                    if name.endswith('_other')
                ]
                df = df.drop(columns=otherNames, errors='ignore').copy()
                for name in otherNames:
                    column = name[:-len('_other')]
                    if column not in df:
                        df[column] = 1.
                for column in ('frame_i', 'Cell_ID'):
                    if column not in df and column not in df.index.names:
                        df[column] = 0
            # Preview rows still use the whole table as other-cell candidates.
            df, formula, alternatives = (
                self._predictors.OtherCellMeasurements.prepare_inputs(
                    df, formula, self.otherCells, self.SizeT,
                    max_targets=10 if preview else None,
                    return_alternatives=True
                )
            )
        else:
            alternatives = None
        return super().evaluateEquation(
            df, formula, enforce_dtype_output=enforce_dtype_output,
            alternatives=alternatives
        )

    @timepoint_debug_trace
    def validateEquation(self, formula):
        try:
            if not formula.strip():
                return True, formula
            df = self.getTestData(preview=True)
            self.evaluateEquation(
                df, formula, enforce_dtype_output=bool, preview=True
            )
        except Exception as err:
            return False, str(err)
        return True, formula

    @timepoint_debug_trace
    def test_cb(self):
        formula = self.formulaEditWidget.text()
        success, message = self.validateEquation(formula)
        if success and formula.strip():
            try:
                self.evaluateEquation(
                    self.getTestData(), formula, enforce_dtype_output=bool
                )
            except Exception as err:
                success, message = False, str(err)
        if success:
            self.equation = formula
            self.statusLabel.setText(
                'Filter equation is valid.'
                if formula.strip() else 'No filter equation; all cells are kept.'
            )
            self.testOutputDisplay.setPlainText(self.statusLabel.text())
        else:
            self.statusLabel.setText(message)
            self.testOutputDisplay.setPlainText(message)
        self.testOutputDisplay.setStyleSheet(
            f'border: 2px solid {"green" if success else "red"}'
        )
        self.okButton.setEnabled(success)

    def ok_cb(self):
        # A passing Test already evaluated the current equation on the whole
        # table (edits disable OK again): do not evaluate it twice
        if not self.okButton.isEnabled():
            self.test_cb()
            if not self.okButton.isEnabled():
                return
        self.equation = self.formulaEditWidget.text().strip()
        self.cancel = False
        self.close()

    def showHelp(self):
        txt = html_utils.paragraph(
            'Return a boolean condition or numeric 0/1. With the other-cell '
            'inputs in the main-cell filter, use the predictor\'s configured '
            'source and window, e.g. <code>COUNT(cell_area_pxl_other) &gt; 0</code> '
            'with Daughter cells and match Any to retain whole cells with '
            'daughters. The predictor still sees all their frames. With the other-cell '
            'filter, use <code>this_</code> before an input name to reference '
            'the enquiring cell at frame t. Express "if" as a condition, e.g. '
            '<code>(area &gt; 5) &amp; (cell_cycle_stage == "S")</code>. '
            'Use <code>&amp;</code>, <code>|</code>, <code>~</code> and '
            'parentheses instead of Python if/else statements or and/or. '
            '<code>SHIFT(area, 1)</code> is the previous observed value of '
            'a cell. To select candidates in the previous video frame, set '
            'both frame-window bounds to <code>t - 1</code>.'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, 'Cell filter equation', txt)


class TimepointIdentifierDialog(QBaseDialog):
    """Set name, color, shortcut and predictor of a time point identifier.

    After closing, `cancel` is False if accepted and `identifier` is a dict
    with keys 'name', 'color', 'shortcut', 'predictor' and 'autoApply'.

    `existingNames` (names that cannot be used) and `reservedShortcuts`
    ({normalized key sequence: name of the action using it}) can also be
    callables returning them: OK then checks the current values, so two
    dialogs open at the same time (the dialog is not modal) cannot both
    add the same name or shortcut.
    """
    def __init__(
            self, name, color, shortcut, predictor=None,
            existingNames=(), reservedShortcuts=None, acdc_df=None,
            SizeT=None, allChNames=(), isZstack=False, isSegm3D=False,
            title='Add time point identifier', parent=None
        ):
        self._predictors = timepoint_predictors
        super().__init__(parent)
        self.setWindowTitle(title)
        self.cancel = True
        self.identifier = None
        self._existingNames = existingNames
        self._reservedShortcuts = reservedShortcuts
        self.acdc_df = acdc_df
        self.SizeT = SizeT
        self.allChNames = list(allChNames)
        self.isZstack = isZstack
        self.isSegm3D = isSegm3D
        if predictor is None:
            predictor = {'type': timepoint_predictors.PREDICTOR_NONE}
        self.equation = predictor.get('equation', '')
        self.equationMode = predictor.get('mode', 'first_true')
        self.otherCells = predictor.get('other_cells')
        self.mainCellFilter = predictor.get('main_cell_filter')
        
        formLayout = QFormLayout()
        self.nameLineEdit = QLineEdit(name)
        formLayout.addRow('Name:', self.nameLineEdit)
        
        self.colorButton = widgets.myColorButton(parent=parent, color=color)
        self.colorButton.setToolTip('Color of the time point lines')
        formLayout.addRow('Color:', self.colorButton)
        
        self.shortcutLineEdit = widgets.ShortcutLineEdit(allowModifiers=True)
        self.shortcutLineEdit.setText(shortcut)
        self.shortcutLineEdit.setToolTip(
            'Click and press a key (combination). Press Backspace to clear.\n\n'
            'Hover over the measurements plot and press the shortcut\n'
            'to add a time point at the frame under the mouse cursor.'
        )
        formLayout.addRow('Shortcut:', self.shortcutLineEdit)
        
        # auto apply on new cells 
        self.autoApplyCheckBox = widgets.Toggle(label_text='Auto apply on unvisited cells',
                                                parent=self)
        self.autoApplyCheckBox.setToolTip(
            'Evaluate the predictor on each cell when it is displayed for '
            'the first time.'
        )
        formLayout.addRow('', self.autoApplyCheckBox)
        
        self.predictorCombobox = QComboBox()
        self.predictorCombobox.addItem(
            'None (manual annotation)', timepoint_predictors.PREDICTOR_NONE
        )
        self.predictorCombobox.addItem(
            'Equation', timepoint_predictors.PREDICTOR_EQUATION
        )
        self.predictorCombobox.addItem(
            'Select own function (Python file)', 
            timepoint_predictors.PREDICTOR_FILE
        )
        self.predictorCombobox.setToolTip(
            'Predict the time points of each cell.\n\n'
            'With "Auto apply on unvisited cells", each cell is predicted when it\n'
            'is displayed for the first time; cells that were already viewed are\n'
            'not predicted. To predict all cells, right-click the identifier\n'
            'action and re-evaluate its predictor, or use "Re-evaluate predictors\n'
            'on all cells" in the toolbar (Shift+shortcut: displayed cell).\n\n'
            'Predicted time points can be moved or removed afterwards.'
        )
        formLayout.addRow('Predictor:', self.predictorCombobox)
        
        self.equationWidget = QWidget()
        equationLayout = QGridLayout(self.equationWidget)
        equationLayout.setContentsMargins(0, 0, 0, 0)
        self.equationLineEdit = QLineEdit(self.equation)
        self.equationLineEdit.setReadOnly(True)
        self.equationLineEdit.setPlaceholderText('No equation set')
        self.equationModeLabel = QLabel()
        self.editEquationButton = widgets.calcPushButton('Edit equation...')
        equationLayout.addWidget(self.equationLineEdit, 0, 0)
        equationLayout.addWidget(self.editEquationButton, 0, 1)
        equationLayout.addWidget(self.equationModeLabel, 1, 0, 1, 2)
        formLayout.addRow('', self.equationWidget)
        
        self.fileWidget = QWidget()
        fileLayout = QGridLayout(self.fileWidget)
        fileLayout.setContentsMargins(0, 0, 0, 0)
        self.fileLineEdit = QLineEdit(predictor.get('filepath', ''))
        self.fileLineEdit.setPlaceholderText('Python file with predictor(df)')
        self.browseFileButton = widgets.browseFileButton(
            'Browse...', ext={'Python': '.py'}, 
            title='Select Python file with predictor(df)'
        )
        self.saveTemplateButton = widgets.savePushButton('Save template...')
        fileInfo = QLabel(html_utils.paragraph(
            'The file must define <code>predictor(df)</code>. '
            '<code>df</code> is the table of <b>one cell</b> (one row per '
            'frame, with <code>frame_i</code> and <code>Cell_ID</code> '
            'columns). Return a <code>frame_i</code> (0-based), a list of '
            '<code>frame_i</code>, or <code>None</code>.<br>'
            'Optionally, add the keyword argument <code>entire_df</code> '
            '(<code>predictor(df, entire_df=None)</code>) to also receive '
            'the table of all cells.', font_size='11px'
        ))
        fileInfo.setWordWrap(True)
        fileLayout.addWidget(self.fileLineEdit, 0, 0)
        fileLayout.addWidget(self.browseFileButton, 0, 1)
        fileLayout.addWidget(self.saveTemplateButton, 0, 2)
        fileLayout.addWidget(fileInfo, 1, 0, 1, 3)
        formLayout.addRow('', self.fileWidget)
        
        self.statusLabel = QLabel()
        self.statusLabel.setWordWrap(True)
        
        buttonsLayout = QHBoxLayout()
        self.testButton = widgets.calcPushButton('Test predictor')
        cancelButton = widgets.cancelPushButton('Cancel')
        okButton = widgets.okPushButton(' Ok ')
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(self.testButton)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addWidget(okButton)
        
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(formLayout)
        mainLayout.addWidget(self.statusLabel)
        mainLayout.addLayout(buttonsLayout)
        self.setLayout(mainLayout)
        self.setFont(fonts.font)
        
        index = self.predictorCombobox.findData(predictor.get('type'))
        self.predictorCombobox.setCurrentIndex(max(index, 0))
        self.predictorChanged()
        self.updateEquationModeLabel()
        
        self.predictorCombobox.currentIndexChanged.connect(self.predictorChanged)
        self.editEquationButton.clicked.connect(self.editEquation)
        self.browseFileButton.sigPathSelected.connect(self.fileLineEdit.setText)
        self.saveTemplateButton.clicked.connect(self.saveTemplate)
        self.testButton.clicked.connect(self.testPredictor)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
    
    def predictorType(self):
        return self.predictorCombobox.currentData()
    
    def predictorChanged(self, *args):
        kind = self.predictorType()
        self.equationWidget.setVisible(
            kind == self._predictors.PREDICTOR_EQUATION
        )
        self.fileWidget.setVisible(kind == self._predictors.PREDICTOR_FILE)
        self.testButton.setVisible(kind != self._predictors.PREDICTOR_NONE)
        self.statusLabel.clear()
    
    def updateEquationModeLabel(self):
        mode = self._predictors.EQUATION_MODES.get(self.equationMode, '')
        self.equationModeLabel.setText(f'Time point(s) of each cell: {mode}')
    
    def editEquation(self):
        win = TimepointPredictorEquationDialog(
            self.allChNames, self.isZstack, self.isSegm3D, 
            acdc_df=self.acdc_df, name=self.nameLineEdit.text(),
            equation=self.equation, mode=self.equationMode,
            otherCells=self.otherCells,
            mainCellFilter=self.mainCellFilter,
            SizeT=self.SizeT, parent=self
        )
        validator = getattr(self, 'validatePredictorDependencies', None)
        if validator is not None:
            win.validatePredictorDependencies = (
                lambda predictor: validator(self.nameLineEdit.text().strip(), predictor)
            )
        with disabled_while_nested(self):
            win.exec_()
        # Deferred: its result is read below
        win.deleteLater()
        if win.cancel:
            return
        self.equation = win.equation
        self.equationMode = win.mode
        self.otherCells = win.otherCells
        self.mainCellFilter = win.mainCellFilter
        self.equationLineEdit.setText(self.equation)
        self.updateEquationModeLabel()
        self.statusLabel.clear()
    
    def saveTemplate(self):
        filepath = QFileDialog.getSaveFileName(
            self, 'Save predictor template', 'timepoint_predictor.py',
            'Python (*.py)'
        )[0]
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(self._predictors.PREDICTOR_FILE_TEMPLATE)
        except OSError as err:
            self.warn('Saving template failed', html.escape(str(err)))
            return
        self.fileLineEdit.setText(filepath)
    
    def predictor(self):
        kind = self.predictorType()
        if kind == self._predictors.PREDICTOR_EQUATION:
            predictor = {
                'type': kind, 'equation': self.equation, 
                'mode': self.equationMode
            }
            if self.otherCells is not None:
                predictor['other_cells'] = dict(self.otherCells)
            if self.mainCellFilter is not None:
                predictor['main_cell_filter'] = dict(self.mainCellFilter)
            return predictor
        if kind == self._predictors.PREDICTOR_FILE:
            return {'type': kind, 'filepath': self.fileLineEdit.text().strip()}
        return {'type': kind}
    
    def validatePredictor(self):
        predictor = self.predictor()
        self._predictors.TimepointPredictor.validate(predictor)
        validator = getattr(self, 'validatePredictorDependencies', None)
        if validator is not None:
            validator(self.nameLineEdit.text().strip(), predictor)
        if predictor['type'] == self._predictors.PREDICTOR_FILE:
            self._predictors.TimepointPredictor.load_function(predictor['filepath'])
        return predictor
    
    @timepoint_debug_trace
    def testPredictor(self):
        try:
            predictor = self.validatePredictor()
            if self.acdc_df is None or self.acdc_df.empty:
                self.statusLabel.setText(
                    'Predictor is valid. Load measurements to test it on data.'
                )
                return
            timepoints = self._predictors.TimepointPredictor.predict(
                self.acdc_df, predictor, SizeT=self.SizeT
            )
        except Exception as err:
            traceback.print_exc()
            self.statusLabel.setText(html_utils.paragraph(
                f'<font color="red">{html.escape(str(err))}</font>'
            ))
            return False
        numCells = self.acdc_df.index.get_level_values('Cell_ID').nunique()
        numTimepoints = sum(len(frames) for frames in timepoints.values())
        self.statusLabel.setText(
            f'Predicted {numTimepoints} time point(s) for {len(timepoints)} '
            f'of {numCells} cells.'
        )
        return True
    
    def warn(self, title, text):
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, title, html_utils.paragraph(text))

    def existingNames(self):
        names = self._existingNames
        return set(names() if callable(names) else names or ())

    def reservedShortcuts(self):
        reserved = self._reservedShortcuts
        return dict((reserved() if callable(reserved) else reserved) or {})

    def ok_cb(self):
        name = self.nameLineEdit.text().strip()
        if not name:
            self.warn('Empty name', 'The name of the time point cannot be empty.')
            return
        if name in self.existingNames():
            self.warn(
                'Name already used', 
                'A time point identifier called '
                f'"{html.escape(name)}" already exists.'
            )
            return
        shortcut = self.shortcutLineEdit.text().strip()
        if shortcut:
            if shortcut.startswith('Mouse '):
                self.warn('Invalid shortcut', 'Mouse buttons are not supported.')
                return
            shortcut = QKeySequence(shortcut).toString()
            usedBy = self.reservedShortcuts().get(shortcut)
            if usedBy is not None:
                self.warn(
                    'Shortcut already used', 
                    f'The shortcut "{html.escape(shortcut)}" is already used '
                    f'by "{html.escape(usedBy)}". Choose a different shortcut.'
                )
                return
        try:
            predictor = self.validatePredictor()
        except Exception as err:
            self.warn('Invalid predictor', html.escape(str(err)))
            return

        self.autoApply = self.autoApplyCheckBox.isChecked()
        self.identifier = {
            'name': name,
            'color': tuple(self.colorButton.color().getRgb()[:3]),
            'shortcut': shortcut,
            'predictor': predictor,
            'autoApply': self.autoApply,
        }
        self.cancel = False
        self.close()


# cellacdc-refactor: intermediate compat block (removed in C12)
# Names that moved out of this module stay reachable here with a
# DeprecationWarning (cellacdc/_compat.py, see CODEMAP.md).
from cellacdc import _compat as _acdc_compat  # noqa: E402

__getattr__, __dir__ = _acdc_compat.moved_module(__name__)
