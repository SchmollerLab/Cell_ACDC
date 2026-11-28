"""Time point to phase translator dialog and its rule dialog (library:
measurements_viewer/phase_translator.py).
"""
import html
import traceback
from functools import partial
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QScrollArea, QVBoxLayout, QWidget
)
from cellacdc import html_utils
from .. import phase_translator
from cellacdc import qutils, urls, widgets
from cellacdc.timepoint_predictors import TIMEPOINT_PREFIX
from . import predictor_editors as viewer_dialogs_predictor_editors


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
        self.predictorGroupbox = viewer_dialogs_predictor_editors._CustomAnnotationPredictorGroupbox(
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

    def chosenPhaseColors(self):
        """User-chosen colors {name: '#rrggbb'} of the current phases (what
        the viewer stores): defaults depend on the phase order, so a stored
        default could be repeated by the default of a phase added later."""
        names = phase_translator.phase_names(
            self.rules, self.annotationOverrides
        )
        return {
            name: self.phaseColors[name] for name in names
            if name in self.phaseColors
        }

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
