"""Conditional formatting dialogs of the measurements viewer columns and
their color helpers (relative colors such as "lighter 150").
"""
import copy
import html
import numpy as np
import pandas as pd
import pyqtgraph as pg
import re
from functools import partial
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QToolButton,
    QVBoxLayout
)
from cellacdc import apps, html_utils
from cellacdc import widgets
from . import predictor_editors as viewer_dialogs_predictor_editors


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
        # The rules cannot be removed, moved or edited while the editor runs
        with viewer_dialogs_predictor_editors.disabled_while_nested(self):
            condition = self.editCondition(
                self.rows[i]['edit'].text().strip()
            )
        if condition is not None and i < len(self.rows):
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


class _FormattingConditionDialog(viewer_dialogs_predictor_editors._TimepointCellFilterEquationDialog):
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
        from cellacdc import timepoint_predictors
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
            '<code>(frame_i &gt;= MAX(frame_i * timepoint_G2_start)) &amp; '
            'any(timepoint_G2_start)</code> (from the G2_start time point '
            'on, only in cells that have it). '
            'Use <code>&amp;</code>, <code>|</code>, <code>~</code> and '
            'parentheses to combine conditions.'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, 'Conditional formatting condition', txt)
