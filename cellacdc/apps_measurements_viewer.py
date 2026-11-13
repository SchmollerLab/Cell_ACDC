"""Dialogs and widgets used only by the measurements viewer
(`measurements_viewer.MeasurementsViewerWin`).

Moved out of `apps.py` and `measurements_viewer.py`; shared dialogs
(e.g. the custom annotation and combine-measurements equation editors)
stay in `apps.py`.
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
from . import (
    apps,
    background_values,
    fonts,
    html_utils,
    myutils,
    phase_translator,
    qutils,
    settings_folderpath,
    timepoint_predictors,
    urls,
    widgets,
)
from .timepoint_predictors import TIMEPOINT_PREFIX, debug_trace as timepoint_debug_trace
from .apps import (
    PreProcessParamsWidget,
    PreProcessRecipeDialogUtil,
    QBaseDialog,
    _CombineMetricsDialog,
)


class _CustomAnnotationPredictorGroupbox(QGroupBox):
    """Predictor settings of a custom annotation (measurements viewer). 
    
    `options` is a dict with optional keys 'acdc_df', 'SizeT', 'allChNames', 
    'isZstack' and 'isSegm3D' used to edit and test the equation.
    """
    def __init__(self, options, predictor=None, autoApply=False, parent=None):
        from . import custom_annotation_predictors, timepoint_predictors
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
        win.exec_()
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
        from . import timepoint_predictors
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
                self.mainCellFilterPresets.update({
                    'Cells with parent': 'parent_ID_tree != -1',
                })
        mainCellCondition = (
            '' if self._mainCellFilter is None
            else self._mainCellFilter['condition']
        )
        self._mainCellCustomCondition = mainCellCondition
        self.mainCellFilterCombobox = QComboBox()
        mainCellPreset = 'custom'
        for label, condition in self.mainCellFilterPresets.items():
            self.mainCellFilterCombobox.addItem(label, label)
            if condition == mainCellCondition:
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
        mainCellMode = (
            'any' if self._mainCellFilter is None
            else self._mainCellFilter['condition_mode']
        )
        self.mainCellMatchCombobox.setCurrentIndex(
            self.mainCellMatchCombobox.findData(mainCellMode)
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
            self.equationChanged
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
        condition = (
            self._mainCellCustomCondition if preset == 'custom'
            else self.mainCellFilterPresets[preset]
        )
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
        settings = {
            'condition': (
                self.mainCellConditionEdit.text().strip() if preset == 'custom'
                else self.mainCellFilterPresets[preset]
            ),
            'condition_mode': self.mainCellMatchCombobox.currentData(),
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
        dialog.exec_()
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
        dialog.exec_()
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
            # trees) but are valid per-cell inputs; SizeT is a constant.
            usesIndexInputs = bool(
                {'frame_i', 'Cell_ID'}
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
                if otherName not in self.otherMeasurementColumns:
                    return False, f'Unknown other-cell input "{otherName}".'
                column = self.otherMeasurementColumns[otherName]
                if self.acdcDf is not None and column not in available:
                    return False, f'Unknown input column "{column}".'
            if (
                    self._predictors._OTHER_FUNCTION_PATTERN.search(formula)
                    or otherMeasurementCalls
                ):
                for match in self._predictors._OTHER_FUNCTION_PATTERN.finditer(formula):
                    column = match.group('column').strip('`')
                    if self.acdcDf is not None and column not in available:
                        return False, f'Unknown input column "{column}".'
                formula = self._predictors._OTHER_FUNCTION_PATTERN.sub(
                    '1', formula
                )
                formula = self._predictors._OTHER_MEASUREMENT_PATTERN.sub(
                    '1', formula
                )
            self.evaluateEquation(self.getTestData(preview=True), formula)
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
        from . import custom_annotation_predictors
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
        from . import timepoint_predictors
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


class ChannelPlaceholderMappingDialog(apps.QBaseDialog):
    """Map each channel placeholder of a premade configuration to a channel
    and enter the values of its input placeholders.

    `placeholders` maps placeholder names to descriptions; `channels` are the
    available channel names. `inputs` are the template's validated
    `input_placeholders` (see `MeasurementsViewerWin.validateInputPlaceholders`),
    each shown with its default value and an info button for its `info` text.
    `validate(mapping, values)` (optional, e.g. resolving the template)
    may raise ValueError to refuse a selection: its message is shown and
    the dialog stays open.
    `channelHints` (`{placeholder: [name parts]}`, optional) improve the
    preselected channels: each placeholder gets an unused channel whose name
    contains the placeholder name, else the first of its hints (in order,
    case-insensitive), else the first unused channel.
    `backgroundValues` is the template's validated `background_values`
    section (`MeasurementsViewerWin.validateBackgroundValuesSpec`): its
    linked inputs can be filled from a summary table of the "Compute
    background intensity" utility, loaded from a file (start folder
    `tableStartDir`) or computed by
    `computeBackgroundValues(channels, backgroundValues, parent=dialog)`,
    which returns the summary table (None if cancelled). The table row of
    each linked channel placeholder is the row of its selected channel
    (the user picks another row when the table has none). Changing the
    channel of a linked placeholder after filling refills its values from
    the same table. While the background values are computed, the dialog
    is disabled.
    After `exec_`, `mapping` is `{placeholder: channel}` and `values` is
    `{input name: typed value}` unless `cancel` is True.
    """

    def __init__(
            self, placeholders, channels, parent=None, inputs=None,
            validate=None, backgroundValues=None,
            computeBackgroundValues=None, tableStartDir='', channelHints=None
        ):
        super().__init__(parent)
        self.cancel = True
        self.mapping = None
        self.values = None
        self.validate = validate
        self.inputs = dict(inputs or {})
        self.backgroundValues = backgroundValues
        self.computeBackgroundValues = computeBackgroundValues
        self.tableStartDir = tableStartDir
        self.backgroundSourceLabel = None
        # Last table used to fill the values: {'summary', 'source', 'mapping'}
        self.backgroundFill = None
        self._closed = False
        self.setWindowTitle('Set up premade configuration')
        channels = sorted(channels)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(html_utils.paragraph(
            'This premade configuration uses placeholder channels.<br>'
            'Select the channel of your experiment for each of them:'
        )))
        grid = QGridLayout()
        self.comboboxes = {}
        guesses = self.guessChannels(placeholders, channels, channelHints)
        for row, (name, description) in enumerate(placeholders.items()):
            text = f'<b>{name}</b>'
            if description:
                text += f'<br><i>{description}</i>'
            label = QLabel(html_utils.paragraph(text))
            combobox = QComboBox()
            combobox.addItems(channels)
            combobox.setCurrentText(guesses[name])
            grid.addWidget(label, row, 0)
            grid.addWidget(combobox, row, 1)
            self.comboboxes[name] = combobox
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        self.inputLineEdits = {}
        if self.inputs:
            layout.addSpacing(10)
            layout.addWidget(QLabel(html_utils.paragraph(
                'Enter the values requested by the configuration:'
            )))
            inputsGrid = QGridLayout()
            for row, (name, spec) in enumerate(self.inputs.items()):
                text = f'<b>{name}</b>'
                if spec.get('description'):
                    text += f'<br><i>{spec["description"]}</i>'
                inputsGrid.addWidget(QLabel(html_utils.paragraph(text)), row, 0)
                lineEdit = QLineEdit(str(spec['default']))
                lineEdit.setToolTip(f'Type: {spec["type"]}')
                inputsGrid.addWidget(lineEdit, row, 1)
                if spec.get('info'):
                    infoButton = widgets.infoPushButton(
                        info_text=html_utils.paragraph(spec['info']),
                        info_title=name
                    )
                    inputsGrid.addWidget(infoButton, row, 2)
                self.inputLineEdits[name] = lineEdit
            inputsGrid.setColumnStretch(1, 1)
            layout.addLayout(inputsGrid)
            if self.backgroundValues is not None:
                layout.addLayout(self._backgroundValuesLayout())
        layout.addSpacing(10)
        buttons = widgets.CancelOkButtonsLayout()
        buttons.okButton.clicked.connect(self.ok_cb)
        buttons.cancelButton.clicked.connect(self.close)
        layout.addLayout(buttons)
        self.setLayout(layout)

    @staticmethod
    def guessChannels(placeholders, channels, channelHints=None):
        """Preselected channel of each placeholder: an unused channel whose
        name contains the placeholder name, else the one containing the
        first matching hint of `channelHints[placeholder]`, else the first
        unused channel (case-insensitive)."""
        channelHints = channelHints or {}
        guesses = dict.fromkeys(placeholders)

        def unused(ch):
            return ch not in guesses.values()

        # Placeholder names first (most specific), then the hints
        for name in placeholders:
            guesses[name] = next(
                (ch for ch in channels
                 if name.lower() in ch.lower() and unused(ch)),
                None
            )
        for name in placeholders:
            if guesses[name] is not None:
                continue
            for hint in channelHints.get(name, []):
                guesses[name] = next(
                    (ch for ch in channels
                     if hint.lower() in ch.lower() and unused(ch)),
                    None
                )
                if guesses[name] is not None:
                    break
        for name in placeholders:
            if guesses[name] is None:
                guesses[name] = next(
                    (ch for ch in channels if unused(ch)),
                    channels[0] if channels else ''
                )
        return guesses

    def currentMapping(self):
        return {
            name: combobox.currentText()
            for name, combobox in self.comboboxes.items()
        }

    def _backgroundValuesLayout(self):
        spec = self.backgroundValues
        layout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()
        if self.computeBackgroundValues is not None:
            computeButton = widgets.calcPushButton(
                'Calculate background values...'
            )
            computeButton.setToolTip(
                'Run the "Compute background intensity" utility on the '
                'Positions of\nthis experiment with the settings of this '
                f'configuration (background\n{spec["statistic"]}, '
                f'{spec["low_percentile"]:g}th and '
                f'{spec["high_percentile"]:g}th percentiles) and fill in '
                'the values'
            )
            computeButton.clicked.connect(self.computeBackgroundValuesClicked)
            buttonsLayout.addWidget(computeButton)
            self.computeBackgroundButton = computeButton
        loadButton = widgets.LoadPushButton(
            'Load calculated background values...'
        )
        loadButton.setToolTip(
            'Fill in the values from the summary '
            'table saved by the "Compute background\n'
            'intensity" utility (<name>_summary.csv in the experiment folder)'
        )
        loadButton.clicked.connect(self.loadBackgroundValuesTable)
        buttonsLayout.addWidget(loadButton)
        self.loadBackgroundTableButton = loadButton
        buttonsLayout.addStretch(1)
        layout.addLayout(buttonsLayout)
        self.backgroundSourceLabel = QLabel()
        self.backgroundSourceLabel.setWordWrap(True)
        self.backgroundSourceLabel.hide()
        layout.addWidget(self.backgroundSourceLabel)
        linked = {link['channel'] for link in spec['inputs'].values()}
        for placeholder in linked:
            self.comboboxes[placeholder].currentTextChanged.connect(
                self.linkedChannelChanged
            )
        return layout

    def linkedBackgroundMapping(self):
        mapping = self.currentMapping()
        return {
            link['channel']: mapping[link['channel']]
            for link in self.backgroundValues['inputs'].values()
        }

    def linkedChannelChanged(self, text):
        """Refill the values from the last table when the channel of a
        linked placeholder changes (they belong to the previous channel)."""
        fill = self.backgroundFill
        if fill is None or self.linkedBackgroundMapping() == fill['mapping']:
            return
        self.fillBackgroundValues(fill['summary'], fill['source'])

    def closeEvent(self, event):
        self._closed = True
        super().closeEvent(event)

    def linkedChannels(self):
        """Selected channels of the placeholders linked to background
        values (each once, in link order)."""
        mapping = self.currentMapping()
        return list(dict.fromkeys(
            mapping[link['channel']]
            for link in self.backgroundValues['inputs'].values()
        ))

    def _warnBackgroundValues(self, title, text):
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, title, html_utils.paragraph(text))

    def computeBackgroundValuesClicked(self):
        # The selection dialogs are not modal: keep this dialog from being
        # used (OK, Cancel, a second run) until the computation returns
        self.setEnabled(False)
        try:
            summary = self.computeBackgroundValues(
                self.linkedChannels(), self.backgroundValues, parent=self
            )
        finally:
            self.setEnabled(True)
        if summary is None or self._closed:
            return
        if summary.empty:
            self._warnBackgroundValues(
                'No background values',
                'No Position was processed, no value was filled in.'
            )
            return
        self.fillBackgroundValues(summary, 'the computed summary table')

    def loadBackgroundValuesTable(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Select summary table of the background values',
            self.tableStartDir, 'CSV files (*.csv);;All files (*)'
        )
        if not filepath:
            return
        try:
            summary = background_values.read_summary_table(filepath)
        except ValueError as err:
            self._warnBackgroundValues(
                'Invalid table',
                f'{err}<br><br>File: <code>{html.escape(filepath)}</code>'
            )
            return
        self.fillBackgroundValues(
            summary, f'<code>{html.escape(filepath)}</code>'
        )

    def askTableChannel(self, placeholder, channel, tableChannels):
        """Ask which row (channel) of the table to use for `placeholder`,
        mapped to `channel` that the table does not have. Returns the
        channel or None to skip its values."""
        win = widgets.QDialogListbox(
            'Select channel of the table',
            f'The table has no values for the channel "{channel}" '
            f'({placeholder}).\n'
            'Select the channel of the table to use instead:',
            tableChannels, multiSelection=False, parent=self,
            allowEmptySelection=False
        )
        win.exec_()
        if win.cancel or not win.selectedItemsText:
            return None
        return win.selectedItemsText[0]

    def fillBackgroundValues(self, summary, source):
        """Fill the inputs linked to background values from a summary table
        (`background_values.read_summary_table`); `source` (HTML) is shown
        in the dialog. Values that are missing (NaN) are left unchanged.
        Returns `{input name: value}` of the filled inputs."""
        spec = self.backgroundValues
        mapping = self.currentMapping()
        rows = {
            str(row['channel']): row for _, row in summary.iterrows()
        }
        tableRows = {}
        for link in spec['inputs'].values():
            placeholder = link['channel']
            if placeholder in tableRows:
                continue
            channel = mapping[placeholder]
            if channel not in rows:
                channel = self.askTableChannel(
                    placeholder, channel, list(rows)
                )
            tableRows[placeholder] = channel
        filled, empty, skipped = {}, [], []
        for name, link in spec['inputs'].items():
            channel = tableRows[link['channel']]
            if channel is None:
                skipped.append(name)
                continue
            try:
                value = float(rows[channel][link['value']])
            except (TypeError, ValueError):
                value = np.nan
            if not np.isfinite(value):
                empty.append(
                    f'<b>{html.escape(name)}</b> (<code>'
                    f'{html.escape(link["value"])}</code> of '
                    f'{html.escape(channel)})'
                )
                continue
            self.inputLineEdits[name].setText(f'{value:.10g}')
            filled[name] = value
        self.backgroundFill = {
            'summary': summary, 'source': source,
            'mapping': self.linkedBackgroundMapping(),
        }
        notes = self._backgroundSettingsNotes(
            {ch: rows[ch] for ch in tableRows.values() if ch is not None},
            uses_percentiles=any(
                link['value'] != 'constant_background'
                for link in spec['inputs'].values()
            )
        )
        used = ', '.join(
            f'{html.escape(placeholder)} = {html.escape(channel)}'
            for placeholder, channel in tableRows.items()
            if channel is not None
        )
        self.backgroundSourceLabel.setText(html_utils.paragraph(
            f'Filled in {len(filled)} of {len(spec["inputs"])} values from '
            f'{source}' + (f' (table rows: {used})' if used else '') + '.',
            font_size='11px'
        ))
        self.backgroundSourceLabel.show()
        if empty or skipped or notes:
            items = []
            if empty:
                items.append(
                    'The table has <b>no value</b> for: '
                    f'{", ".join(empty)}. Min/max values are empty when no '
                    'acdc_output table with the mean intensity of the '
                    'channel was selected.'
                )
            if skipped:
                items.append(
                    'Not filled in (no table row selected): '
                    + ', '.join(f'<b>{html.escape(n)}</b>' for n in skipped)
                )
            items.extend(notes)
            self._warnBackgroundValues(
                'Background values',
                f'Filled in {len(filled)} of {len(spec["inputs"])} values. '
                'The other values are left unchanged.'
                + html_utils.to_list(items)
            )
        return filled

    def _backgroundSettingsNotes(self, rows, uses_percentiles=True):
        """Differences between the settings of the table rows and of the
        configuration (statistic, percentiles and ignored zero pixels)."""
        spec = self.backgroundValues
        notes = []
        for channel, row in rows.items():
            differences = []
            statistic = str(row['background_statistic'])
            if statistic != spec['statistic']:
                differences.append(
                    f'background {html.escape(statistic)} '
                    f'(configuration: {spec["statistic"]})'
                )
            low, high = row['low_percentile'], row['high_percentile']
            if uses_percentiles and not (
                    np.isclose(low, spec['low_percentile'])
                    and np.isclose(high, spec['high_percentile'])):
                differences.append(
                    f'{low:g}th/{high:g}th percentiles (configuration: '
                    f'{spec["low_percentile"]:g}th/'
                    f'{spec["high_percentile"]:g}th)'
                )
            excludeZero = configparser.ConfigParser.BOOLEAN_STATES.get(
                str(row.get('exclude_zero_pixels', '')).strip().lower()
            )
            if (excludeZero is not None
                    and excludeZero != spec['exclude_zero_pixels']):
                ignored = 'ignored' if excludeZero else 'included'
                expected = (
                    'ignored' if spec['exclude_zero_pixels'] else 'included'
                )
                differences.append(
                    f'pixels with intensity 0 {ignored} '
                    f'(configuration: {expected})'
                )
            if differences:
                notes.append(
                    f'The values of {html.escape(str(channel))} were computed '
                    f'with {" and ".join(differences)}.'
                )
        return notes

    @staticmethod
    def parseInput(text, kind):
        """Convert the entered `text` to `kind` ('float', 'int' or 'str')."""
        if kind == 'str':
            return text
        text = text.strip()
        if kind == 'int':
            return int(text)
        value = float(text)
        if not np.isfinite(value):
            raise ValueError('not a finite number')
        return value

    def currentValues(self):
        """Return `{name: typed value}`; raise ValueError on invalid input."""
        values = {}
        for name, lineEdit in self.inputLineEdits.items():
            kind = self.inputs[name]['type']
            try:
                values[name] = self.parseInput(lineEdit.text(), kind)
            except ValueError:
                raise ValueError(
                    f'"{html.escape(lineEdit.text())}" is not a valid {kind} '
                    f'for <b>{html.escape(name)}</b>.'
                )
        return values

    def ok_cb(self):
        try:
            inputValues = self.currentValues()
        except ValueError as err:
            msg = widgets.myMessageBox(wrapText=False)
            msg.warning(
                self, 'Invalid value', html_utils.paragraph(str(err))
            )
            return
        mapping = self.currentMapping()
        values = list(mapping.values())
        duplicated = sorted({ch for ch in values if values.count(ch) > 1})
        if duplicated:
            msg = widgets.myMessageBox(wrapText=False)
            msg.warning(
                self, 'Channel used twice',
                html_utils.paragraph(
                    'Each placeholder must be mapped to a different channel. '
                    'Used more than once: '
                    f'<b>{html.escape(", ".join(duplicated))}</b>'
                )
            )
            return
        if self.validate is not None:
            try:
                self.validate(mapping, inputValues)
            except ValueError as err:
                msg = widgets.myMessageBox(wrapText=False)
                msg.warning(
                    self, 'Invalid selection',
                    html_utils.paragraph(html.escape(str(err)))
                )
                return
        self.mapping = mapping
        self.values = inputValues
        self.cancel = False
        self.close()


# Conditional formatting colors: '#rrggbb', or a shade of the column color,
# 'lighter'/'darker' with an optional QColor factor in percent (>= 100),
# e.g. 'darker:250'.
RELATIVE_COLOR_PATTERN = re.compile(r'(lighter|darker)(?::(\d+))?')
RELATIVE_COLOR_FACTORS = {'lighter': 150, 'darker': 200}


def is_formatting_color(spec):
    if not isinstance(spec, str):
        return False
    if re.fullmatch(r'#[0-9a-fA-F]{6}', spec):
        return True
    match = RELATIVE_COLOR_PATTERN.fullmatch(spec)
    return bool(match) and (match[2] is None or int(match[2]) >= 100)


def formatting_color(spec, baseColor):
    """QColor of a formatting color `spec` for a column of `baseColor`."""
    match = RELATIVE_COLOR_PATTERN.fullmatch(spec)
    if match is None:
        return QColor(spec)
    factor = int(match[2]) if match[2] else RELATIVE_COLOR_FACTORS[match[1]]
    base = pg.mkColor(baseColor)
    return base.lighter(factor) if match[1] == 'lighter' else base.darker(factor)


class ConditionalFormattingDialog(apps.QBaseDialog):
    """Edit the ordered conditional formatting rules of a plotted column.

    `rules` is a list of `{'condition': str, 'color': spec}` where spec is
    '#rrggbb' or a shade of the column color ('lighter', 'darker', optionally
    with a factor such as 'darker:250', see `formatting_color`).
    `validateCondition(condition)` raises if a condition cannot be evaluated;
    `editCondition(condition)` opens an equation editor and returns the new
    condition (None if canceled). After `exec_`, `rules` holds the edited
    rules unless `cancel` is True.
    """

    # Okabe-Ito colors (colorblind-friendly) without red, which is reserved
    # for the current frame line
    DEFAULT_COLORS = ('#009e73', '#0072b2', '#e69f00', '#cc79a7', '#56b4e9')

    def __init__(
            self, column, rules, validateCondition=None, editCondition=None,
            parent=None
        ):
        super().__init__(parent)
        self.cancel = True
        self.column = column
        self.rules = copy.deepcopy(rules)
        self.validateCondition = validateCondition
        self.editCondition = editCondition
        self.setWindowTitle(f'Conditional formatting of "{column}"')
        layout = QVBoxLayout()
        layout.addWidget(QLabel(html_utils.paragraph(
            f'Color the time course of <b>{column}</b> where a condition is '
            'true.<br>Conditions are equations of the selected cell\'s '
            'measurements (bool or 0/1, one value per frame or one for the '
            'whole cell),<br>e.g. <code>mCherry_ramp == 1</code> or '
            '<code>area &gt; 1.5 * MEAN(area)</code>. Per-cell functions '
            '(MAX, DIFF, SHIFT, any, ...) and <code>SizeT</code> are '
            'available.<br>Rules are checked in order and the first true '
            'rule colors the point and the line to the next point.<br>'
            'Frames where no rule is true keep the column color. Use a '
            'lighter or darker shade of the column color, or a custom color.'
        )))
        self.rowsLayout = QGridLayout()
        self.rowsLayout.setColumnStretch(3, 1)
        layout.addLayout(self.rowsLayout)
        addButton = widgets.addPushButton(' Add rule')
        addButton.clicked.connect(self.addRule)
        addLayout = QHBoxLayout()
        addLayout.addWidget(addButton)
        addLayout.addStretch(1)
        layout.addLayout(addLayout)
        layout.addSpacing(10)
        buttons = widgets.CancelOkButtonsLayout()
        buttons.okButton.clicked.connect(self.ok_cb)
        buttons.cancelButton.clicked.connect(self.close)
        layout.addLayout(buttons)
        self.setLayout(layout)
        self.rows = []
        self.rebuildRows()
        self.resize(max(self.width(), 650), self.height())

    def syncRules(self):
        self.rules = [
            {
                'condition': row['edit'].text().strip(),
                'color': (
                    row['mode'].currentData()
                    if row['mode'].currentData() != 'custom'
                    else row['color'].color(mode='qcolor').name()
                ),
            }
            for row in self.rows
        ]

    def rebuildRows(self):
        for row in self.rows:
            for widget in row.values():
                self.rowsLayout.removeWidget(widget)
                widget.deleteLater()
        self.rows = []
        for i, rule in enumerate(self.rules):
            spec = rule['color']
            match = RELATIVE_COLOR_PATTERN.fullmatch(spec)
            mode = QComboBox()
            mode.setToolTip(
                'Shade of the column color, or a custom color'
            )
            for text, data in (('Lighter shade', 'lighter'),
                               ('Darker shade', 'darker'),
                               ('Custom color', 'custom')):
                # Keep a custom factor (e.g. 'darker:250') of the rule
                if match and match[1] == data:
                    data = spec
                mode.addItem(text, data)
            mode.setCurrentIndex(
                ('lighter', 'darker').index(match[1]) if match else 2
            )
            colorButton = pg.ColorButton(
                color=spec if match is None else self.DEFAULT_COLORS[0]
            )
            colorButton.setToolTip('Custom color of the points and lines')
            colorButton.setEnabled(match is None)
            mode.currentIndexChanged.connect(
                lambda idx, button=colorButton: button.setEnabled(idx == 2)
            )
            edit = QLineEdit(rule['condition'])
            edit.setPlaceholderText('Condition, e.g. measurement_ramp == 1')
            row = {
                'label': QLabel(f'{i+1}.'), 'mode': mode,
                'color': colorButton, 'edit': edit
            }
            if self.editCondition is not None:
                row['editor'] = widgets.editPushButton('Edit...')
                row['editor'].setToolTip('Open the equation editor')
                row['editor'].clicked.connect(partial(self.editRule, i))
            for name, text, tooltip, step in (
                    ('up', '↑', 'Move up (higher priority)', -1),
                    ('down', '↓', 'Move down (lower priority)', 1)):
                button = QToolButton()
                button.setText(text)
                button.setToolTip(tooltip)
                button.setEnabled(0 <= i + step < len(self.rules))
                button.clicked.connect(partial(self.moveRule, i, step))
                row[name] = button
            row['remove'] = widgets.delPushButton()
            row['remove'].setToolTip('Remove rule')
            row['remove'].clicked.connect(partial(self.removeRule, i))
            for col, widget in enumerate(row.values()):
                self.rowsLayout.addWidget(widget, i, col)
            self.rows.append(row)

    def addRule(self, checked=False):
        self.syncRules()
        used = {rule['color'] for rule in self.rules}
        color = next(
            (c for c in ('lighter', 'darker', *self.DEFAULT_COLORS)
             if c not in used),
            self.DEFAULT_COLORS[len(self.rules) % len(self.DEFAULT_COLORS)]
        )
        self.rules.append({'condition': '', 'color': color})
        self.rebuildRows()
        self.rows[-1]['edit'].setFocus()

    def removeRule(self, i, checked=False):
        self.syncRules()
        self.rules.pop(i)
        self.rebuildRows()

    def moveRule(self, i, step, checked=False):
        self.syncRules()
        rules = self.rules
        rules[i], rules[i + step] = rules[i + step], rules[i]
        self.rebuildRows()

    def editRule(self, i, checked=False):
        condition = self.editCondition(self.rows[i]['edit'].text().strip())
        if condition is not None:
            self.rows[i]['edit'].setText(condition)

    def ok_cb(self, checked=False):
        self.syncRules()
        for i, rule in enumerate(self.rules):
            error = None
            if not rule['condition']:
                error = 'The condition is empty.'
            elif self.validateCondition is not None:
                try:
                    self.validateCondition(rule['condition'])
                except Exception as err:
                    error = str(err)
            if error is None:
                continue
            msg = widgets.myMessageBox(wrapText=False)
            msg.warning(
                self, 'Invalid condition', html_utils.paragraph(
                    f'Rule {i+1} (<code>{html.escape(rule["condition"])}'
                    f'</code>) cannot be evaluated:<br><br>{html.escape(error)}'
                )
            )
            return
        self.cancel = False
        self.close()


class _FormattingConditionDialog(_TimepointCellFilterEquationDialog):
    """Equation editor for one conditional formatting condition.

    Conditions see only the selected cell's rows (see the
    `measurements_viewer` module docstring), so other-cell inputs
    (``*_other``, ``*_OTHER(...)``) are neither offered nor accepted.
    Missing values count as False, as in
    `MeasurementsViewerWin.evaluateConditionalFormatting`.
    """

    OTHER_CELL_INPUTS = False

    INFO = (
        'Return bool or 0/1, either a scalar or one value per frame. The '
        'condition is evaluated on the selected cell across its frames; '
        'frames where it is true are colored. Missing values count as '
        'False.'
    )

    @staticmethod
    def conditionMask(values, n):
        """Boolean mask of length ``n`` from a condition result; missing
        values are False. Single conversion shared by this editor and
        `MeasurementsViewerWin.evaluateConditionalFormatting`, so both
        accept and reject the same conditions."""
        from . import timepoint_predictors
        try:
            if isinstance(values, (pd.Series, pd.Index)):
                values = values.to_numpy(dtype=float, na_value=np.nan)
            elif values is None or values is pd.NA:
                values = np.nan
            values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
        except (TypeError, ValueError):
            raise ValueError(
                'The condition must return bool or 0/1 values.'
            ) from None
        return timepoint_predictors.CellFilter.boolean_values(values, n)

    def evaluateEquation(
            self, df, formula, *, enforce_dtype_output=None, preview=False
        ):
        if enforce_dtype_output is not bool:
            return super().evaluateEquation(
                df, formula, enforce_dtype_output=enforce_dtype_output,
                preview=preview
            )
        values = super().evaluateEquation(df, formula, preview=preview)
        if isinstance(values, pd.Series):
            return pd.Series(
                self.conditionMask(values, len(values)), index=values.index
            )
        return self.conditionMask(values, len(df))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Conditional formatting condition')
        for label in self.findChildren(QLabel):
            if label.text() == 'Filter equation:':
                label.setText('Condition:')
        intro = self.layout().itemAt(0).widget()
        if isinstance(intro, QLabel):
            intro.setText(html_utils.paragraph(self.INFO))

    def validateEquation(self, formula):
        usesOtherCells = (
            any(
                name.endswith('_other')
                for name in self._predictors.CellEquation.names(formula)
            )
            or re.search(r'_OTHER\s*\(', formula, re.IGNORECASE) is not None
        )
        if usesOtherCells:
            return False, (
                'Other-cell inputs (*_other, *_OTHER(...)) are not available '
                'in conditional formatting.'
            )
        return super().validateEquation(formula)

    def showHelp(self):
        txt = html_utils.paragraph(
            f'{self.INFO}<br><br>Examples: <code>mCherry_ramp == 1</code>, '
            '<code>area &gt; 1.5 * MEAN(area)</code>, '
            '<code>frame_i &gt;= MAX(frame_i * timepoint_G2_start)</code>. '
            'Use <code>&amp;</code>, <code>|</code>, <code>~</code> and '
            'parentheses to combine conditions.'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, 'Conditional formatting condition', txt)


class PhaseTranslatorRuleDialog(QDialog):
    """Edit eligibility and the ordered, named intervals of one phase rule."""

    def __init__(
            self, timepoints, annotations, options, rule=None,
            existingNames=(), parent=None
        ):
        super().__init__(parent)
        self.cancel = True
        self.existingNames = set(existingNames)
        self.setWindowTitle('Time point to phase rule editor')
        rule = rule or {
            'name': '', 'predictor': {'type': 'none'},
            'timepoints': [], 'annotations': [], 'phases': ['']
        }
        self.boundaries = list(rule['timepoints'])
        self._phaseNames = dict(zip(
            self.intervalKeys(), rule['phases']
        ))
        self.phaseEdits = {}
        self.nameEdit = QLineEdit(rule['name'])
        options = dict(options, nameGetter=self.nameEdit.text)
        self.predictorGroupbox = _CustomAnnotationPredictorGroupbox(
            options, predictor=rule['predictor'], parent=self
        )
        self.predictorGroupbox.setTitle('Rule eligibility formula')
        self.predictorGroupbox.autoApplyCheckBox.hide()
        self.predictorGroupbox.predictorCombobox.setItemText(
            0, 'Always eligible (no formula)'
        )
        self.predictorGroupbox.predictorCombobox.setToolTip(
            'The formula must pass for the entire cell using All/None/Most.'
        )
        self.requirementChecks = {}
        self.available = set(timepoints) | set(annotations)
        requirements = QWidget()
        requirementsLayout = QVBoxLayout(requirements)
        for title, available, selected, isBoundary in (
            ('Time points (interval boundaries)', timepoints, rule['timepoints'], True),
            ('Custom annotations (requirements only)', annotations, rule['annotations'], False)
        ):
            requirementsLayout.addWidget(QLabel(title))
            for column in dict.fromkeys([*available, *selected]):
                text = (
                    column[len(TIMEPOINT_PREFIX):] if isBoundary else column
                )
                if column not in available:
                    text += ' (missing)'
                checkbox = QCheckBox(text)
                checkbox.setChecked(column in selected)
                self.requirementChecks[column] = (checkbox, isBoundary)
                requirementsLayout.addWidget(checkbox)
                if isBoundary:
                    checkbox.toggled.connect(partial(self.boundaryToggled, column))
        requirementsLayout.addStretch(1)
        requirementsScroll = QScrollArea()
        requirementsScroll.setWidgetResizable(True)
        requirementsScroll.setWidget(requirements)
        intervals = QWidget()
        self.intervalsLayout = QVBoxLayout(intervals)
        intervalsScroll = QScrollArea()
        intervalsScroll.setWidgetResizable(True)
        intervalsScroll.setWidget(intervals)
        columns = QHBoxLayout()
        columns.addWidget(requirementsScroll)
        columns.addWidget(intervalsScroll)
        buttons = QHBoxLayout()
        cancelButton = widgets.cancelPushButton('Cancel')
        self.okButton = widgets.okPushButton('  OK  ')
        buttons.addStretch(1)
        buttons.addWidget(cancelButton)
        buttons.addWidget(self.okButton)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Rule name:'))
        layout.addWidget(self.nameEdit)
        layout.addWidget(self.predictorGroupbox)
        info = QLabel(
            'All checked inputs must be present in the cell history. Each '
            'boundary must occur exactly once, in this order. Phase intervals '
            'include their start but exclude their end. Empty names leave '
            'intervals unassigned.'
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addLayout(columns, stretch=1)
        layout.addLayout(buttons)
        cancelButton.clicked.connect(self.reject)
        self.okButton.clicked.connect(self.ok_cb)
        self.rebuildIntervals()
        self.resize(950, 800)

    def intervalKeys(self):
        edges = [None, *self.boundaries, None]
        return list(zip(edges, edges[1:]))

    def rememberPhaseNames(self):
        self._phaseNames.update({
            key: edit.text() for key, edit in self.phaseEdits.items()
        })

    def boundaryToggled(self, column, checked):
        self.rememberPhaseNames()
        if checked:
            self.boundaries.append(column)
        else:
            self.boundaries.remove(column)
        self.rebuildIntervals()

    def moveBoundary(self, column, offset):
        self.rememberPhaseNames()
        index = self.boundaries.index(column)
        other = index + offset
        if not 0 <= other < len(self.boundaries):
            return
        self.boundaries[index], self.boundaries[other] = (
            self.boundaries[other], self.boundaries[index]
        )
        self.rebuildIntervals()

    def rebuildIntervals(self):
        while self.intervalsLayout.count():
            item = self.intervalsLayout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.phaseEdits = {}
        for index, key in enumerate(self.intervalKeys()):
            edit = QLineEdit(self._phaseNames.get(key, ''))
            edit.setPlaceholderText(
                'Phase before first point' if index == 0 and self.boundaries else
                'Phase after last point' if index == len(self.boundaries) and self.boundaries else
                'Phase between points' if self.boundaries else 'Phase for entire cell'
            )
            self.phaseEdits[key] = edit
            self.intervalsLayout.addWidget(edit)
            if index == len(self.boundaries):
                break
            column = self.boundaries[index]
            row = QWidget()
            rowLayout = QHBoxLayout(row)
            rowLayout.setContentsMargins(0, 0, 0, 0)
            rowLayout.addWidget(QLabel(column[len(TIMEPOINT_PREFIX):]), stretch=1)
            for label, offset, buttonClass in (
                    ('Up', -1, widgets.arrowUpPushButton),
                    ('Down', 1, widgets.arrowDownPushButton)):
                button = buttonClass(label)
                button.setEnabled(0 <= index + offset < len(self.boundaries))
                button.clicked.connect(
                    partial(self.moveBoundary, column, offset)
                )
                rowLayout.addWidget(button)
            self.intervalsLayout.addWidget(row)
        self.intervalsLayout.addStretch(1)

    def ok_cb(self):
        try:
            name = self.nameEdit.text().strip()
            if name in self.existingNames:
                raise ValueError('A phase rule with this name already exists.')
            required = [
                column for column, (check, _) in self.requirementChecks.items()
                if check.isChecked()
            ]
            missing = set(required).difference(self.available)
            if missing:
                raise ValueError('Uncheck missing inputs: ' + ', '.join(sorted(missing)))
            predictor = self.predictorGroupbox.validatePredictor()
            self.rule = {
                'name': name, 'predictor': predictor,
                'timepoints': list(self.boundaries),
                'annotations': [
                    column for column, (check, boundary)
                    in self.requirementChecks.items()
                    if check.isChecked() and not boundary
                ],
                'phases': [
                    self.phaseEdits[key].text().strip() for key in self.intervalKeys()
                ],
            }
            phase_translator.validate_rules([self.rule])
        except Exception as err:
            traceback.print_exc()
            widgets.myMessageBox(wrapText=False).warning(
                self, 'Invalid phase rule',
                html_utils.paragraph(html.escape(str(err)))
            )
            return
        self.cancel = False
        self.accept()


PHASE_TRANSLATOR_HELP = (
    'The translator turns the time points of each cell into <b>phases</b>: '
    'all frames between two consecutive time points get the same phase '
    'name (e.g., "G1" between "Birth" and "S start").'
    '<br><br><b>How to use it:</b>'
    + html_utils.to_list([
        'Define time point identifiers in the main window (<b>Add time point '
        'identifier...</b>) and mark them on the cells (hover over the plot '
        'and press their shortcut, or use their predictors).',
        'Click <b>Add...</b> to create a rule: check the time points that '
        'separate the phases, put them in their order (boundaries) and type '
        'the phase names before, between and after them. A blank name '
        'leaves those frames unassigned. Optionally add a formula and the '
        'custom annotations that a cell must have for the rule to apply.',
        'Add more rules for cells with other time points (e.g., cells '
        'present only at the end) and order them with <b>Up</b>/<b>Down</b>: '
        'for each cell the first rule (from the top) whose formula and '
        'requirements pass supplies its phases. Each checked time point '
        'must occur exactly once in the cell, in the boundary order.',
        'Custom annotations with the option <b>Overrides phases</b> (set it '
        'when adding or editing the annotation) take precedence over the '
        'rules: an annotated cell gets the annotation name as its phase on '
        'every frame (e.g., "Arrested" or "Excluded"). If a cell has '
        'several of them, the first annotation in the toolbar wins. The '
        'rules apply to the other cells.',
        'Click <b>Apply phase evaluation</b> on the displayed cell or on all '
        'cells, or check the automatic evaluation to update the phases '
        'whenever time points or annotations change.',
        'The phases are drawn as a colored band below the time course and '
        'color the Measurement vs Measurement points (change the colors in '
        '<b>Phase colors</b>). <b>Export measurements...</b> saves the '
        '<code>phase</code> and <code>phase_&lt;name&gt;</code> columns.',
    ], ordered=True)
    + 'Save the viewer configuration to keep the rules and colors.'
)


class PhaseTranslatorDialog(QDialog):
    """Manage phase rules in descending priority, without editing the table.

    ``annotationOverrides`` are the names of the custom annotations with
    the "Overrides phases" option (toolbar order). They are not edited
    here: the dialog lists them, gives them color buttons and allows
    applying the evaluation without rules.
    """
    sigApply = Signal(object, bool)

    def __init__(
            self, rules, timepoints, annotations, options, parent=None,
            phaseColors=None, annotationOverrides=()
        ):
        import copy
        super().__init__(parent)
        self.cancel = True
        self.rules = copy.deepcopy(rules)
        self.annotationOverrides = list(annotationOverrides)
        # User-chosen colors {phase name: '#rrggbb'}; others use defaults
        self.phaseColors = dict(phaseColors or {})
        self.phaseColorButtons = {}
        self.timepoints = timepoints
        self.annotations = annotations
        self.options = options
        self.setWindowTitle('Time point to phase translator')
        self.rulesList = QListWidget()
        self.rulesList.itemDoubleClicked.connect(lambda *_: self.editRule())
        controls = QHBoxLayout()
        for text, callback, buttonClass in (
            ('Add...', self.addRule, widgets.addPushButton),
            ('Edit...', self.editRule, widgets.editPushButton),
            ('Remove...', self.removeRule, widgets.delPushButton),
            ('Up', partial(self.moveRule, -1), widgets.arrowUpPushButton),
            ('Down', partial(self.moveRule, 1), widgets.arrowDownPushButton)
        ):
            button = buttonClass(text)
            button.clicked.connect(callback)
            controls.addWidget(button)
        self.autoApplyCheckbox = QCheckBox(
            'Automatically evaluate phases when time points or annotations '
            'change or phases are missing'
        )
        self.autoApplyCheckbox.setToolTip(
            'Re-evaluate the phases of a cell whenever '
            'its time points are added, moved,\n'
            'removed or predicted or its custom '
            'annotations change (annotated, removed\n'
            'or predicted), and evaluate displayed cells without phases. Applied with\n'
            'Apply or OK; save the viewer configuration to keep this option.'
        )
        applyButtons = QHBoxLayout()
        # Same icons as the "Apply" / "Apply to all Positions..." buttons
        # of the post-processing dialog in apps.py
        self.applyCellButton = widgets.okPushButton(
            'Apply phase evaluation on displayed cell', isDefault=False
        )
        self.applyAllButton = widgets.futurePushButton(
            'Apply phase evaluation on all cells'
        )
        for button, allCells in (
            (self.applyCellButton, False), (self.applyAllButton, True)
        ):
            button.clicked.connect(lambda checked=False, batch=allCells: self.apply(batch))
            applyButtons.addWidget(button)
        buttons = QHBoxLayout()
        self.helpButton = widgets.infoPushButton('Help...')
        self.helpButton.setToolTip(
            'How to translate time points into phases (with a link to the '
            'user guide)'
        )
        self.helpButton.clicked.connect(self.showHelp)
        cancelButton = widgets.cancelPushButton('Cancel')
        okButton = widgets.okPushButton('  OK  ')
        buttons.addWidget(self.helpButton)
        buttons.addStretch(1)
        buttons.addWidget(cancelButton)
        buttons.addWidget(okButton)
        cancelButton.clicked.connect(self.reject)
        okButton.clicked.connect(self.ok_cb)
        layout = QVBoxLayout(self)
        info = QLabel(
            'Rules are evaluated from top to bottom for each cell. The first '
            'rule whose formula and all requirements pass supplies its phases '
            '(cells with a custom annotation that overrides phases get the '
            'annotation name instead). '
            'Apply updates the current position in memory; Export measurements '
            'saves the phase and phase_<name> columns. Save the viewer '
            'configuration to keep these rules.'
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        # Custom annotations with "Overrides phases" (edited with the
        # annotation, not here)
        self.overridesLabel = QLabel()
        self.overridesLabel.setWordWrap(True)
        self.overridesLabel.setTextFormat(Qt.RichText)
        if self.annotationOverrides:
            names = ', '.join(
                f'<b>{html.escape(name)}</b>'
                for name in self.annotationOverrides
            )
            firstWins = (
                '; the first one in this list wins'
                if len(self.annotationOverrides) > 1 else ''
            )
            self.overridesLabel.setText(
                f'Cells annotated with {names} get that annotation as their '
                'phase on every frame (custom annotation option "Overrides '
                f'phases"{firstWins}); the rules apply to the other cells.'
            )
        self.overridesLabel.setVisible(bool(self.annotationOverrides))
        layout.addWidget(self.overridesLabel)
        layout.addWidget(self.rulesList)
        layout.addLayout(controls)
        self.phaseColorsGroupbox = QGroupBox('Phase colors')
        self.phaseColorsGroupbox.setToolTip(
            'Colors of the phase band below the time course and of the\n'
            'Measurement vs Measurement points. Click a color to change it.'
        )
        self.phaseColorsLayout = QGridLayout(self.phaseColorsGroupbox)
        layout.addWidget(self.phaseColorsGroupbox)
        layout.addWidget(self.autoApplyCheckbox)
        layout.addLayout(applyButtons)
        layout.addLayout(buttons)
        self.refreshRules()
        self.resize(800, 450)

    def showHelp(self, checked=False):
        """How-to of the translator with a link to its user guide section
        (`urls.measurements_viewer_phase_translator_docs_url`)."""
        msg = widgets.myMessageBox()
        msg.information(
            self, 'Time point to phase translator: help',
            html_utils.paragraph(PHASE_TRANSLATOR_HELP),
            url_to_open=urls.measurements_viewer_phase_translator_docs_url,
            open_url_button_text='Open user guide'
        )

    def currentPhaseColors(self):
        """Colors {name: '#rrggbb'} of the current rules' phases, then of
        the annotations overriding phases."""
        return phase_translator.phase_colors(
            phase_translator.phase_names(
                self.rules, self.annotationOverrides
            ),
            self.phaseColors
        )

    def refreshPhaseColors(self):
        while self.phaseColorsLayout.count():
            widget = self.phaseColorsLayout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self.phaseColorButtons = {}
        colors = self.currentPhaseColors()
        columns = 4
        for i, (name, color) in enumerate(colors.items()):
            button = widgets.myColorButton(color=color)
            toolTip = f'Color of phase "{name}"'
            if name in self.annotationOverrides:
                toolTip += ' (custom annotation overriding phases)'
            button.setToolTip(qutils.wrap_tooltip(toolTip))
            button.sigColorChanged.connect(
                partial(self.phaseColorChanged, name)
            )
            row, col = divmod(i, columns)
            # Color first, then the phase name (legend-like)
            self.phaseColorsLayout.addWidget(button, row, 2*col)
            self.phaseColorsLayout.addWidget(QLabel(name), row, 2*col + 1)
            self.phaseColorButtons[name] = button
        self.phaseColorsGroupbox.setVisible(bool(colors))

    def phaseColorChanged(self, name, button):
        self.phaseColors[name] = button.color().name()

    def refreshRules(self, row=0):
        self.rulesList.clear()
        self.rulesList.addItems([
            f'{index + 1}. {rule["name"]}'
            for index, rule in enumerate(self.rules)
        ])
        self.rulesList.setCurrentRow(row)
        # Annotations overriding phases can be evaluated without rules
        canApply = bool(self.rules or self.annotationOverrides)
        self.applyCellButton.setEnabled(canApply)
        self.applyAllButton.setEnabled(canApply)
        self.refreshPhaseColors()

    def ruleDialog(self, rule=None):
        dialog = PhaseTranslatorRuleDialog(
            self.timepoints, self.annotations, self.options, rule=rule,
            existingNames=[
                item['name'] for item in self.rules if item is not rule
            ], parent=self
        )
        dialog.exec_()
        result = None if dialog.cancel else dialog.rule
        dialog.deleteLater()
        return result

    def addRule(self, checked=False):
        rule = self.ruleDialog()
        if rule is not None:
            self.rules.append(rule)
            self.refreshRules(len(self.rules) - 1)

    def editRule(self, checked=False):
        row = self.rulesList.currentRow()
        if row < 0:
            return
        rule = self.ruleDialog(self.rules[row])
        if rule is not None:
            self.rules[row] = rule
            self.refreshRules(row)

    def removeRule(self, checked=False):
        row = self.rulesList.currentRow()
        if row >= 0:
            self.rules.pop(row)
            self.refreshRules(min(row, len(self.rules) - 1))

    def moveRule(self, offset, checked=False):
        row = self.rulesList.currentRow()
        other = row + offset
        if row < 0 or not 0 <= other < len(self.rules):
            return
        self.rules[row], self.rules[other] = self.rules[other], self.rules[row]
        self.refreshRules(other)

    def apply(self, allCells):
        import copy
        self.sigApply.emit(copy.deepcopy(self.rules), allCells)

    def ok_cb(self):
        self.cancel = False
        self.accept()


class TimepointIdentifierDialog(QBaseDialog):
    """Set name, color, shortcut and predictor of a time point identifier.

    After closing, `cancel` is False if accepted and `identifier` is a dict
    with keys 'name', 'color', 'shortcut', 'predictor' and 'autoApply'.
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
        self.existingNames = set(existingNames)
        # {normalized key sequence: name of the action using it}
        self.reservedShortcuts = dict(reservedShortcuts or {})
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
        win.exec_()
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
    
    def ok_cb(self):
        name = self.nameLineEdit.text().strip()
        if not name:
            self.warn('Empty name', 'The name of the time point cannot be empty.')
            return
        if name in self.existingNames:
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
            usedBy = self.reservedShortcuts.get(shortcut)
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


class MeasurementProcessingParamsWidget(PreProcessParamsWidget):
    recipesPath = os.path.join(settings_folderpath, 'measurement_processing_recipes')
    recipePrefix = 'measurement_processing_recipe'

    def createSelector(self):
        return widgets.SignalProcessingSelector()

    @staticmethod
    def castRecipeValue(param, value):
        """Convert a recipe value (e.g. a string read from a saved INI
        recipe) to the type of the ``run()`` parameter ``param``.

        Booleans are parsed with ``configparser.ConfigParser.BOOLEAN_STATES``
        because ``bool('False')`` is True. Raises ``ValueError`` for values
        that cannot be converted.
        """
        if not isinstance(value, str):
            return param.type(value)
        if param.type is bool:
            try:
                return configparser.ConfigParser.BOOLEAN_STATES[
                    value.strip().lower()
                ]
            except KeyError:
                raise ValueError(
                    f'Invalid boolean value {value!r} for "{param.name}".'
                ) from None
        if hasattr(param.type, 'cast_dtype'):
            return param.type.cast_dtype(value)
        return param.type(value)

    def recipe(self, warn=True):
        recipe = []
        for step in self.stepsWidgets.values():
            selector = step['selector']
            method = selector.currentText()
            function = selector.preprocessMapper[method]['function']
            saved = selector.methodToDefaultValuesMapper.get(method, {})
            kwargs = {
                param.name: self.castRecipeValue(
                    param, saved.get(param.name, param.default)
                )
                for param in myutils.get_function_argspec(function)
            }
            kwargs.update(step.get('step_kwargs', {}))
            recipe.append({'method': method, 'kwargs': kwargs})
        return recipe

    def loadRecipe(self, configPars):
        from .measurement_processing import MEASUREMENT_PROCESSING_MAPPER

        methods = [section.get('method') for section in configPars.values()]
        if any(method not in MEASUREMENT_PROCESSING_MAPPER for method in methods):
            msg = widgets.myMessageBox(wrapText=False)
            msg.warning(
                self, 'Invalid measurement recipe',
                'This recipe contains unsupported measurement-processing steps.'
            )
            return
        # Saved INI recipes store every value as a string: convert them to
        # the run() parameter types so that both recipe() and the parameters
        # dialog (askSetParams) get typed values (e.g. 'False' -> False).
        typedConfigPars = {}
        try:
            for section, section_items in configPars.items():
                method = section_items.get('method')
                function = MEASUREMENT_PROCESSING_MAPPER[method]['function']
                params = {
                    param.name: param
                    for param in myutils.get_function_argspec(function)
                }
                typedItems = {}
                for option, value in section_items.items():
                    param = params.get(option)
                    if option != 'method' and param is not None:
                        value = self.castRecipeValue(param, value)
                    typedItems[option] = value
                typedConfigPars[section] = typedItems
        except (TypeError, ValueError) as err:
            msg = widgets.myMessageBox(wrapText=False)
            msg.warning(
                self, 'Invalid measurement recipe',
                'This recipe contains invalid parameter values:<br><br>'
                f'{html.escape(str(err))}'
            )
            return
        super().loadRecipe(typedConfigPars)
        for step in self.stepsWidgets.values():
            step.pop('step_kwargs', None)


class MeasurementProcessingRecipeDialog(PreProcessRecipeDialogUtil):
    """Select measurement leaves by channel/type, with recursive fuzzy search."""
    paramsWidgetClass = MeasurementProcessingParamsWidget
    selectionLabel = 'Select measurement(s) to process:'
    emptySuffixMessage = 'Enter a suffix for the output column(s).'

    def __init__(
            self, measurement_names, parent=None,
            acdc_df=None, allChNames=()
        ):
        self._measurementDataframe = acdc_df
        self._measurementChannels = allChNames
        super().__init__(measurement_names, parent=parent)
        self.measurement_names = measurement_names
        self.measurementsSearchLineEdit = QLineEdit()
        self.measurementsSearchLineEdit.setPlaceholderText('Search measurement...')
        self.measurementsSearchLineEdit.setClearButtonEnabled(True)
        self.measurementsSearchLineEdit.textChanged.connect(self.filterMeasurements)
        self.mainLayout.insertWidget(
            self.mainLayout.indexOf(self.listSelector),
            self.measurementsSearchLineEdit
        )
        self.setWindowTitle('Measurement-processing recipe')
        self.preProcessParamsWidget.groupbox.setTitle('Measurement processing')
        self.keepInputDataTypeToggle.setChecked(False)

        self.append_new_colname = widgets.alphaNumericLineEdit(parent=self)
        self.append_new_colname.setPlaceholderText('Append to column for saving...')
        self.mainLayout.insertWidget(
            self.mainLayout.indexOf(self.keepInputDataTypeLayout),
            self.append_new_colname
        )
        self.append_new_colname_tip = QLabel('')
        self.append_new_colname.textChanged.connect(
            self.append_new_colname_changed
        )
        self.listSelector.selectionModel().selectionChanged.connect(
            lambda: self.append_new_colname_changed(self.append_new_colname.text())
        )
        self.append_new_colname_changed(self.append_new_colname.text())
        self.mainLayout.insertWidget(
            self.mainLayout.indexOf(self.append_new_colname) + 1,
            self.append_new_colname_tip
        )

        for i in range(self.keepInputDataTypeLayout.count()):
            widget = self.keepInputDataTypeLayout.itemAt(i).widget()
            if widget is not None:
                widget.hide()
                
    def get_new_colnames(self):
        """Return ``(warn, display, full_list)`` for the output column names.

        ``warn`` is True when the names cannot be used: no suffix was
        entered for the selected measurement(s) (outputs would overwrite
        the inputs) or an output column already exists. ``display`` is
        then the message to show to the user.
        """
        warn, display = False, ''
        text = self.append_new_colname.text()
        
        selected_channels = [index.data() for index in self.listSelector.selectedIndexes()]
        if text and selected_channels:
            full_list = [f"{selected_channel}_{text}" for selected_channel in selected_channels]
            if len(selected_channels) > 1:
                diplay_channel_part = ', '.join(selected_channels)
                display = f'"{diplay_channel_part}"_{text}'
            else:
                display = f'{selected_channels[0]}_{text}'
        elif selected_channels:
            full_list = selected_channels
            return True, self.emptySuffixMessage, full_list
        elif text:
            full_list = [text]
            display = text
        else:
            full_list = []
            display = ""
                    
        forbidden_names = [file_name for file_name in full_list if file_name in self.measurement_names]
        if forbidden_names:
            display = (
                'Output column(s) already exist: '
                + ', '.join(forbidden_names) + '. Choose another suffix.'
            )
            warn = True
            return warn, display, full_list

        warn = False
        return warn, display, full_list

    def append_new_colname_changed(self, text):
        warn, tip, _ = self.get_new_colnames()
        self.append_new_colname_tip.setText(tip)
        # A missing suffix is a neutral hint, not an error
        if warn and tip != self.emptySuffixMessage:
            self.append_new_colname_tip.setStyleSheet("color: red;")
        else:
            self.append_new_colname_tip.setStyleSheet("")

    def createListSelector(self, names):
        names = list(names)
        dataframe = self._measurementDataframe
        if dataframe is None:
            dataframe = pd.DataFrame(columns=names, dtype=float)
        self.measurementsModel = widgets.MeasurementTreeModel(self)
        self.measurementsModel.populate(dataframe[names], self._measurementChannels)
        self.measurementsSearchProxy = widgets.MeasurementTreeSearchProxy(self)
        self.measurementsSearchProxy.setSourceModel(self.measurementsModel)
        selector = widgets.QTreeView()
        selector.setHeaderHidden(True)
        selector.setUniformRowHeights(True)
        selector.setModel(self.measurementsSearchProxy)
        selector.setEditTriggers(QAbstractItemView.NoEditTriggers)
        selector.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.measurementsHighlightDelegate = widgets._ButtonSearchHighlightDelegate(
            selector
        )
        selector.setItemDelegate(self.measurementsHighlightDelegate)
        selector.expandAll()
        first = next(self.measurementsModel.measurementItems(), None)
        if first is not None:
            selector.setCurrentIndex(
                self.measurementsSearchProxy.mapFromSource(first.index())
            )
        return selector

    def filterMeasurements(self, text):
        """Filter and highlight measurements with the viewer's fuzzy search."""
        self.measurementsSearchProxy.setFilterText(text)
        self.measurementsHighlightDelegate.setFilterText(text)
        if text.strip():
            self.listSelector.expandAll()
        self.listSelector.viewport().update()

    def warnChannelSelectionEmpty(self):
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'No measurement selected', html_utils.paragraph(
                'You did <b>not select any measurement</b>.<br><br>'
                'Select at least one measurement to process.'
            )
        )

    def ok_cb(self):
        warn, display, full_list = self.get_new_colnames()
        if warn:
            msg = widgets.myMessageBox(wrapText=False)
            msg.warning(self, 'Invalid output column name', display)
            return
        self.saveNames = full_list
        super().ok_cb()
